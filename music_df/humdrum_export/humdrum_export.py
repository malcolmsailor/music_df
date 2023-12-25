import os
import tempfile
import typing as t
from contextlib import contextmanager

import pandas as pd

from music_df.humdrum_export.collate_spines import collate_spines
from music_df.humdrum_export.color_df import ColorMapping, color_df
from music_df.humdrum_export.constants import DEFAULT_COLOR_MAPPING, USER_SIGNIFIERS
from music_df.humdrum_export.df_to_spines import df_to_spines
from music_df.humdrum_export.df_utils.spell_df import spell_df
from music_df.humdrum_export.df_utils.split_df_by_pitch import split_df_by_pitch
from music_df.humdrum_export.merge_spines import merge_spines


def _write_spine(spine: t.List[str], path: str) -> None:
    with open(path, "w") as outf:
        outf.write("**kern\n")
        for token in spine:
            outf.write(token)
            outf.write("\n")
        outf.write("*-\n")


@contextmanager
def _get_temp_paths(n: int):
    paths = [tempfile.mkstemp(suffix=".krn")[1] for _ in range(n)]
    try:
        yield paths
    finally:
        for path in paths:
            os.remove(path)


POSTSCRIPT = ["!!!filter: autobeam"]


def _check_colors(
    df: pd.DataFrame, color_mapping: ColorMapping | None, uncolored_val: str | None
):
    if "color_char" not in df:
        return POSTSCRIPT

    assert color_mapping is not None
    if not df.color_char.isin(set(USER_SIGNIFIERS) | {""}).all():
        raise ValueError(f"Some items in df.color_char are not in {USER_SIGNIFIERS}")
    postscript = POSTSCRIPT + [
        f'!!!RDF**kern: {signifier} = marked note, color="{color}"'
        for signifier, color in color_mapping.char_to_hex.items()
    ]
    color_key_data = {}
    if uncolored_val is not None:
        color_key_data[uncolored_val] = "#000000"
    for val, color in color_mapping.value_to_hex.items():
        if isinstance(val, tuple):
            # val is tuple of form (value, transparency level)
            assert len(val) == 2
            val = val[0]
        if val in color_key_data:
            assert color_key_data[val] == color[:7]
        else:
            color_key_data[val] = color[:7]
    for val, color in color_key_data.items():
        postscript.append(f"!! color_key: {val}={color}")
    return postscript


def df2hum(
    df: pd.DataFrame,
    # n_clefs: int = 2, # TODO
    split_point: int = 60,
    label_col: t.Optional[str] = None,
    label_mask_col: t.Optional[str] = None,
    label_color_col: t.Optional[str] = None,
    color_col: t.Optional[str] = None,
    color_mask_col: t.Optional[str] = None,
    color_mapping: t.Optional[t.Mapping[t.Any, str]] = None,
    color_transparency_col: t.Optional[str] = None,
    n_transparency_levels: t.Optional[int] = None,
    uncolored_val: t.Optional[str] = None,
) -> str:
    """
    Args:
        df: pandas DataFrame containing a musical score (see `music_df`
            package).
        split_point: int at which to split output into different staves (for
            treble and bass clefs). Default 60.
        label_col: an optional column name. The contents of the columns will be
            used to label the notes.
        color_col: if provided, notes will be colored based on this column.
            The unique values in this column will be mapped to different
            colors, up to 7 different colors, at which point they will
            repeat in modular fashion. If there are values that shouldn't
            be colored (i.e., should be left black), this can be indicated
            with a boolean mask in the column pointed to by the `color_mask_col`
            argument.
        color_transparency_col: if provided, the transparency of notes will
            be indicated with this column. Must be scalar, transparency will be linearly
            interpolated from its low value to its high value.
        color_mapping: optional dictionary mapping values (from color_col) to colors.
            The colors can be indicated w/ hex colors (strings beginning with a "#"
            character). Missing colors are given default values.
            There can be at most 7 distinct colors.
    """
    df = spell_df(df)
    if label_mask_col is not None:
        # (Malcolm 2023-10-20) I'm not sure why we constrain label_color_col to have at
        #   most one color. I think it is so that there is no risk of simultaneous
        #   labels having conflicting colors (since we only have one label
        #   per-time-point and only one color per-label)
        if label_color_col is not None:
            assert len(df.loc[df[label_mask_col], label_color_col].unique()) == 1
        # TODO: (Malcolm 2023-10-20) what do we do if label_color_col is None?
    if color_col is not None:
        color_mapping_inst = ColorMapping(
            df=df,
            color_col=color_col,
            color_mask_col=color_mask_col,
            color_mapping=color_mapping,
            n_alpha_levels=n_transparency_levels,
            uncolored_val=uncolored_val,
        )
        # internal_color_mapping, val_to_color_char = process_color_mapping(
        #     df, color_col, color_mask_col, color_mapping
        # )
        df = color_df(df, color_col, color_transparency_col, color_mapping_inst)
    else:
        color_mapping_inst = None
    postscript = _check_colors(df, color_mapping_inst, uncolored_val)
    # if the last item in the df is not a barline, there can be problems if
    #   after we split the df by pitch, one or more of the dfs is empty in
    #   the last measure (making the final barline seem 1 measure sooner
    #   in those parts)
    if df.iloc[-1].type != "bar":
        last_bar_vals = {"type": ["bar"], "onset": [df.release.max()]}
        if label_mask_col is not None:
            last_bar_vals[label_mask_col] = False
        if color_mask_col is not None:
            last_bar_vals[color_mask_col] = False
        last_bar = pd.DataFrame(last_bar_vals)
        df = pd.concat([df, last_bar])
    dfs = split_df_by_pitch(df, split_point)
    staves = []
    for df in dfs:
        if label_mask_col is not None:
            assert label_color_col is not None
            assert len(df.loc[df[label_mask_col], label_color_col].unique()) == 1
        spines = df_to_spines(
            df,
            label_col=label_col,
            label_mask_col=label_mask_col,
            label_color_col=label_color_col,
        )
        if not spines:
            continue
        with _get_temp_paths(len(spines)) as paths:
            for path, spine in zip(paths, spines):
                _write_spine(spine, path)
            collated = collate_spines(paths)
            merged = merge_spines(collated)
        staves.append(merged)
    assert staves
    with _get_temp_paths(len(staves)) as paths:
        for path, stave in zip(paths, staves):
            with open(path, "w") as outf:
                outf.write("\n".join(stave))
        collated = collate_spines(paths)
    return collated.strip() + "\n" + "\n".join(postscript)


def df2clef(df: pd.DataFrame):
    spines = df_to_spines(df)
    with _get_temp_paths(len(spines)) as paths:
        for path, spine in zip(paths, spines):
            _write_spine(spine, path)
        collated = collate_spines(paths)
    return collated
