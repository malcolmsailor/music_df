import os
import tempfile
import typing as t
from contextlib import contextmanager

import pandas as pd

from music_df.add_feature import add_bar_durs
from music_df.chord_df import label_music_df_with_chord_df, merge_annotations
from music_df.humdrum_export.collate_spines import collate_spines
from music_df.humdrum_export.color_df import ColorMapping, color_df
from music_df.humdrum_export.constants import DEFAULT_COLOR_MAPPING, USER_SIGNIFIERS
from music_df.humdrum_export.df_to_spines import df_to_spines, kern_to_float_dur
from music_df.humdrum_export.df_utils.spell_df import spell_df
from music_df.humdrum_export.df_utils.split_df_by_pitch import split_df_by_pitch
from music_df.humdrum_export.dur_to_kern import dur_to_kern
from music_df.humdrum_export.merge_spines import merge_spines
from music_df.quantize_df import quantize_df
from music_df.sort_df import sort_df
from music_df.split_notes import split_notes_at_barlines


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


def number_measures(humdrum_contents: str):
    """Note: this function assumes that the input has no measure numbers.

    >>> humdrum_contents = '''*C:\\t*C:\\t*C:\\t*C:
    ... *M2/2\\t*M2/2\\t*M2/2\\t*M2/2
    ... *met(c|)\\t*met(c|)\\t*met(c|)\\t*met(c|)
    ... =\\t=\\t=\\t=
    ... 1C\\t1c\\t1G\\t1e
    ... =\\t=\\t=\\t=
    ... 2.B\\t2.d\\t2.e\\t2.g#
    ... 4E\\t4B\\t4d\\t4g
    ... =\\t=\\t=\\t=
    ... 2F\\t2A\\t2c\\t2f
    ... 2F#\\t2B\\t2A\\t2d#
    ... =\\t=\\t=\\t=
    ... 2G\\t2c\\t2G\\t2e
    ... 4C\\t4c\\t4G\\t4e
    ... 4C\\t4c\\t4G\\t4e
    ... =\\t=\\t=\\t=
    ... 4D\\t4c\\t4A\\t4f#
    ... 4D\\t4c\\t4A\\t4f#
    ... 4D\\t4c\\t4A\\t4f#
    ... 4D\\t4c\\t4A\\t4f#
    ... ==\\t==\\t==\\t==
    ... *-\\t*-\\t*-\\t*-
    ... '''

    Visually this doctest appears to be working but I haven't figured out the whitespace
    normalization to get it to pass yet.
    >>> number_measures(humdrum_contents)  # doctest: +SKIP
    '''*C:      *C:     *C:     *C:
    *M2/2        *M2/2   *M2/2   *M2/2
    *met(c|)     *met(c|)        *met(c|)        *met(c|)
    =1   =1      =1      =1
    1C   1c      1G      1e
    =2   =2      =2      =2
    2.B  2.d     2.e     2.g#
    4E   4B      4d      4g
    =3   =3      =3      =3
    2F   2A      2c      2f
    2F#  2B      2A      2d#
    =4   =4      =4      =4
    2G   2c      2G      2e
    4C   4c      4G      4e
    4C   4c      4G      4e
    =5   =5      =5      =5
    4D   4c      4A      4f#
    4D   4c      4A      4f#
    4D   4c      4A      4f#
    4D   4c      4A      4f#
    ==   ==      ==      ==
    *-   *-      *-      *-
    '''
    """
    output_lines = []
    bar_n = 1
    for line in humdrum_contents.split("\n"):
        if not line.startswith("="):
            output_lines.append(line)
            continue
        else:
            bar_tokens = line.split("\t")
        if bar_tokens == ["=="] * len(bar_tokens):
            # Not sure if there should ever be numbered double bars
            output_lines.append(line)
            continue

        assert bar_tokens == ["="] * len(bar_tokens), (
            "bar numbers etc. are not implemented"
        )

        numbered_bars = "\t".join([f"={bar_n}"] * len(bar_tokens))
        output_lines.append(numbered_bars)
        bar_n += 1
    return "\n".join(output_lines)


def _align_pickup_durations(
    spines: t.List[t.List[str]],
) -> t.List[t.List[str]]:
    """Prepend rests to spines with shorter pickups so all have equal duration.

    When a DataFrame is cropped mid-measure, different parts may have different
    amounts of content before the first barline. This causes barline
    misalignment after collation (timebase + assemble), crashing merge_spines.
    """
    if len(spines) <= 1:
        return spines

    pickup_durs: t.List[float] = []
    for spine in spines:
        dur = 0.0
        for token in spine:
            if token == "=":
                break
            if token.startswith("*") or token.startswith("!"):
                continue
            # For chords like "4c 4e", duration comes from the first note
            dur += kern_to_float_dur(token.split(" ")[0])
        pickup_durs.append(dur)

    max_dur = max(pickup_durs)
    if max_dur == 0 or all(abs(d - max_dur) < 1e-9 for d in pickup_durs):
        return spines

    from metricker import Meter

    meter = Meter("4/4")

    aligned: t.List[t.List[str]] = []
    for spine, pickup_dur in zip(spines, pickup_durs):
        diff = max_dur - pickup_dur
        if diff < 1e-9:
            aligned.append(spine)
            continue

        kern_durs = dur_to_kern(diff, offset=0, meter=meter)
        rest_tokens = [f"{k}r" for _, k in kern_durs]

        # Insert rests after any leading metadata tokens (* or !)
        insert_idx = 0
        for token in spine:
            if token.startswith("*") or token.startswith("!"):
                insert_idx += 1
            else:
                break

        aligned.append(spine[:insert_idx] + rest_tokens + spine[insert_idx:])

    return aligned


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
    number_every_nth_note: t.Optional[int] = None,
    number_specified_notes: t.Optional[t.List[int]] = None,
    number_notes_offset: int = 0,
    quantize: None | int = None,
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
    if quantize:
        df = quantize_df(df, tpq=quantize)
    df = add_bar_durs(df)
    df = split_notes_at_barlines(
        df, min_overhang_dur=(1 / 16) if quantize is None else (1 / quantize)
    )

    if number_every_nth_note:
        df["note_index"] = -1
        df.loc[df.type == "note", "note_index"] = range(  # type:ignore
            number_notes_offset,
            (df.type == "note").sum() + number_notes_offset,
        )
        df["nth_note_labels"] = df.note_index.astype(str)
        df.loc[df["note_index"] % number_every_nth_note != 0, "nth_note_labels"] = ""

    if number_specified_notes:
        if "nth_note_labels" not in df.columns:
            df["nth_note_labels"] = ""
        offset_indices = [n + number_notes_offset for n in number_specified_notes]
        mask = df.index.isin(df[df.type == "note"].index[offset_indices])
        df.loc[mask, "nth_note_labels"] = [str(n) for n in number_specified_notes]

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
    for split_df in dfs:
        if label_mask_col is not None:
            assert label_color_col is not None
            assert (
                len(split_df.loc[split_df[label_mask_col], label_color_col].unique())
                == 1
            )
        spines = df_to_spines(
            split_df,
            label_col=label_col,
            label_mask_col=label_mask_col,
            label_color_col=label_color_col,
            nth_note_label_col=(
                "nth_note_labels"
                if (number_every_nth_note or number_specified_notes)
                else None
            ),
        )
        if not spines:
            continue
        spines = _align_pickup_durations(spines)
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
    collated = number_measures(collated)
    return collated.strip() + "\n" + "\n".join(postscript)


def df2clef(df: pd.DataFrame):
    spines = df_to_spines(df)
    with _get_temp_paths(len(spines)) as paths:
        for path, spine in zip(paths, spines):
            _write_spine(spine, path)
        collated = collate_spines(paths)
    return collated


def df_with_harmony_to_hum(
    note_df: pd.DataFrame,
    chord_df: pd.DataFrame,
    split_degree: bool = False,
    include_key: bool = True,
    split_point: int = 60,
    label_color_col: t.Optional[str] = None,
    quantize: t.Optional[int] = None,
    color_col: t.Optional[str] = None,
    color_mask_col: t.Optional[str] = None,
    color_mapping: t.Optional[t.Mapping[t.Any, str]] = None,
    color_transparency_col: t.Optional[str] = None,
    n_transparency_levels: t.Optional[int] = None,
    uncolored_val: t.Optional[str] = None,
) -> str:
    """Export note_df with harmonic annotations from chord_df to Humdrum format.

    Args:
        note_df: DataFrame containing notes.
        chord_df: DataFrame containing chord annotations with onset, key, degree/
            primary_degree+secondary_degree, quality, and inversion columns.
        split_degree: If True, use primary/secondary degree columns instead of
            a single degree column.
        include_key: If True, include key in the annotation labels (e.g., "C.IM").
        split_point: MIDI pitch at which to split into treble/bass staves.
        label_color_col: Optional column for coloring labels.
        quantize: Optional quantization level (ticks per quarter).
        color_col: Optional column for coloring notes.
        color_mask_col: Optional boolean mask column for which notes to color.
        color_mapping: Optional mapping from color_col values to hex colors.
        color_transparency_col: Optional column for note transparency.
        n_transparency_levels: Optional number of transparency levels.
        uncolored_val: Optional value label for uncolored notes in the color key.

    Returns:
        Humdrum-formatted string with harmonic analysis labels.
    """
    if split_degree:
        columns_to_add = (
            "key",
            "primary_degree",
            "primary_alteration",
            "secondary_degree",
            "secondary_alteration",
            "quality",
            "inversion",
        )
    else:
        columns_to_add = ("key", "degree", "quality", "inversion")

    labeled_df = label_music_df_with_chord_df(
        note_df, chord_df, columns_to_add=columns_to_add
    )
    labeled_df = sort_df(labeled_df)
    labeled_df["harmonic_analysis"] = merge_annotations(
        labeled_df, include_key=include_key
    )

    return df2hum(
        labeled_df,
        split_point=split_point,
        label_col="harmonic_analysis",
        label_color_col=label_color_col,
        quantize=quantize,
        color_col=color_col,
        color_mask_col=color_mask_col,
        color_mapping=color_mapping,
        color_transparency_col=color_transparency_col,
        n_transparency_levels=n_transparency_levels,
        uncolored_val=uncolored_val,
    )
