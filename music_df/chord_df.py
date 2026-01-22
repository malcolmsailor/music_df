import io  # noqa: F401
import re
from math import isnan
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from music_df.harmony.chords import CacheDict, get_key_pc_cache, get_rn_pc_cache


def add_chord_pcs(
    chord_df: pd.DataFrame,
    inplace: bool = False,
    rn_pc_cache: CacheDict[tuple[str, str], list[int]] | None = None,
) -> pd.DataFrame:
    """
    Adds a column 'chord_pcs' to the chord_df with the pcs of each chord.

    >>> chord_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... key,rn
    ... b,i
    ... ,V
    ... f#,iv
    ... ,i
    ... '''
    ...     )
    ... )
    >>> add_chord_pcs(chord_df)
      key  rn chord_pcs
    0   b   i       b26
    1   b   V       6a1
    2  f#  iv       b26
    3  f#   i       691
    """
    if not inplace:
        chord_df = chord_df.copy()

    chord_df["key"] = chord_df["key"].ffill()

    if rn_pc_cache is None:
        rn_pc_cache = get_rn_pc_cache(case_matters=True, hex_str=True)

    chord_df["chord_pcs"] = chord_df.apply(
        lambda row: rn_pc_cache[row["rn"], row["key"]],
        axis=1,
    )

    return chord_df


def add_key_pcs(
    key_df: pd.DataFrame,
    inplace: bool = False,
    key_pc_cache: CacheDict[str, list[int] | str] | None = None,
) -> pd.DataFrame:
    """
    Adds a column 'key_pcs' to the df with the pcs of each key.

    >>> key_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... key
    ... b
    ... NaN
    ... f#
    ... '''
    ...     )
    ... )
    >>> add_key_pcs(key_df)
      key  key_pcs
    0   b  b12467a
    1   b  b12467a
    2  f#  689b125
    """
    if not inplace:
        key_df = key_df.copy()

    key_df["key"] = key_df["key"].ffill()

    if key_pc_cache is None:
        key_pc_cache = get_key_pc_cache(hex_str=True)

    key_df["key_pcs"] = key_df["key"].apply(lambda key: key_pc_cache[key])

    return key_df


def get_quality_for_merging(quality: pd.Series | str) -> pd.Series | str:
    # (Malcolm 2024-04-18) possibly we want to do further processing, e.g.
    #   - remove 6 from augmented 6 chords "aug6" quality and otherwise simplify
    #   - only display the quality when it contradicts the expected value for the
    #       scale (this of course would require a lot more coding)
    if isinstance(quality, str):
        return quality.replace("7", "")
    else:
        return quality.str.replace("7", "")


def keep_new_elements_only(
    series: pd.Series, fill_element="", ignore_falsy: bool = True
):
    """
    >>> s = pd.Series(list("aaabbcddde"))
    >>> keep_new_elements_only(s)  # doctest: +NORMALIZE_WHITESPACE
    0    a
    1
    2
    3    b
    4
    5    c
    6    d
    7
    8
    9    e
    dtype: object

    >>> s = pd.Series([float("nan"), "a", "", "a", float("nan"), "b", ""])
    >>> keep_new_elements_only(s)  # doctest: +NORMALIZE_WHITESPACE
    0
    1    a
    2
    3
    4
    5    b
    6
    dtype: object
    """

    out = series.copy()
    if ignore_falsy:
        out[~series.astype(bool)] = float("nan")
    out = out.ffill()

    mask = (out != out.shift(1)) & (~out.isna())
    out[~mask] = fill_element
    return out


def merge_annotations(
    df: pd.DataFrame,
    degree_col: str = "degree",
    primary_degree_col: str = "primary_degree",
    primary_alteration_col: str = "primary_alteration",
    secondary_degree_col: str = "secondary_degree",
    secondary_alteration_col: str = "secondary_alteration",
    inversion_col: str = "inversion",
    quality_col: str = "quality",
    include_key: bool = True,
    key_col: str = "key",
) -> pd.Series:
    """
    We rely on the dataframe being sorted to only show new annotations.

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,key,degree,inversion,quality
    ... bar,,,,
    ... note,C,I,0,M
    ... note,C,V,1,M
    ... bar,na,na,,na
    ... note,C,I,0,M
    ... note,G,V,1,M
    ... '''
    ...     )
    ... )
    >>> merge_annotations(df)  # doctest: +NORMALIZE_WHITESPACE
    0
    1     C.IM
    2      VM6
    3
    4       IM
    5    G.VM6
    dtype: object

    # Not sure what the point, if anything, of the next test is
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,key,degree,inversion,quality
    ... bar,,,,
    ... note,a,V,0,M
    ... note,C,V,0,M
    ... '''
    ...     )
    ... )
    >>> merge_annotations(df)  # doctest: +NORMALIZE_WHITESPACE
    0
    1    a.VM
    2      C.
    dtype: object
    """
    df = df.copy()

    df["inversion_figure"] = df.apply(
        lambda row: inversion_number_to_figure(row[inversion_col], row[quality_col]),
        axis=1,
    )
    df["quality_for_merging"] = get_quality_for_merging(df[quality_col])
    if degree_col not in df.columns:
        assert all(
            col in df.columns
            for col in [
                primary_degree_col,
                primary_alteration_col,
                secondary_degree_col,
                secondary_alteration_col,
            ]
        )
    else:
        df = single_degree_to_split_degrees(
            df,
            degree_col=degree_col,
            inplace=True,
            primary_degree_col=primary_degree_col,
            primary_alteration_col=primary_alteration_col,
            secondary_degree_col=secondary_degree_col,
            secondary_alteration_col=secondary_alteration_col,
        )
    df = split_degrees_to_single_degree(
        df,
        inversion_col="inversion_figure",
        quality_col="quality_for_merging",
        primary_degree_col=primary_degree_col,
        primary_alteration_col=primary_alteration_col,
        secondary_degree_col=secondary_degree_col,
        secondary_alteration_col=secondary_alteration_col,
        output_col="rn",
        inplace=True,
    )
    # I was using ":" as the separator character but it is a special
    #   value in humdrum even when escaped.

    df["rn"] = keep_new_elements_only(df["rn"].replace("na", float("nan")))
    keys = keep_new_elements_only(df[key_col].replace("na", float("nan")) + ".")

    return keys + df["rn"]


def get_unique_annotations_per_onset(
    df: pd.DataFrame,
    annotation_col: str,
    onset_col: str = "onset",
    pitch_col: str = "pitch",
    fill_value: str = "",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Rather than using this function, we can just sort the dataframe and then use
    keep_new_elements_only. This has the virtue of only showing each annotation when
    it first occurs rather than on every new onset.

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,pitch,rn
    ... 0.0,60,IM
    ... 0.0,64,IM
    ... 0.5,62,IM
    ... 1.0,67,VM6
    ... 1.0,59,VM6
    ... '''
    ...     )
    ... )
    >>> get_unique_annotations_per_onset(df, "rn")  # doctest: +NORMALIZE_WHITESPACE
       onset  pitch   rn
    0    0.0     60   IM
    1    0.0     64
    2    0.5     62   IM
    3    1.0     67
    4    1.0     59  VM6
    """
    if not inplace:
        df = df.copy()

    assert annotation_col in df.columns, f"annotation_col {annotation_col} not in df"

    min_pitch_indices = df.groupby(onset_col)[pitch_col].idxmin()
    mask = df.index.isin(min_pitch_indices)
    df.loc[~mask, annotation_col] = fill_value
    return df


TRIAD_INVERSIONS = MappingProxyType({0: "", 1: "6", 2: "64"})
SEVENTH_CHORD_INVERSIONS = MappingProxyType({0: "7", 1: "65", 2: "43", 3: "42"})


def inversion_number_to_figure(
    inversion_number: int,
    quality: str,
    triad_inversions_mapping: Mapping[int, str] | None = None,
    seventh_chord_inversions_mapping: Mapping[int, str] | None = None,
) -> str:
    """
    Convert a number indicating the inversion (0-indexed) to a figured-bass figure.

    Quality is required to distinguish between triads on the one hand and 7th chords and
    augmented 6ths on the other.

    >>> inversion_number_to_figure(0, "M")
    ''
    >>> inversion_number_to_figure(1, "m")
    '6'
    >>> inversion_number_to_figure(0, "m7")
    '7'
    >>> inversion_number_to_figure(3, "Mm7")
    '42'
    """
    # If the chord is a 7th or augmented 6th, we use 7th chord inversions. (Since
    #   we only have integers to indicate 1st, 2nd inversion etc., we can't distinguish
    #   German and Italian 6th chords.)
    inversion_number = float(inversion_number)
    if isnan(inversion_number):
        return ""
    inversion_number = int(inversion_number)
    if "7" in quality or quality == "aug6":
        if seventh_chord_inversions_mapping is None:
            seventh_chord_inversions_mapping = SEVENTH_CHORD_INVERSIONS
        return seventh_chord_inversions_mapping.get(inversion_number, "?")
    # If the quality is unknown we ignore the inversion
    elif quality == "x":
        return ""
    # Otherwise, assume to be a triad
    if triad_inversions_mapping is None:
        triad_inversions_mapping = TRIAD_INVERSIONS
    return triad_inversions_mapping.get(inversion_number, "?")


def split_degrees_to_single_degree(
    df: pd.DataFrame,
    primary_degree_col: str = "primary_degree",
    primary_alteration_col: str = "primary_alteration",
    secondary_degree_col: str = "secondary_degree",
    secondary_alteration_col: str = "secondary_alteration",
    inversion_col: str | None = None,
    quality_col: str | None = None,
    null_alteration_char: str = "_",
    output_col: str = "degree",
    null_chord_token: str = "na",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,primary_degree,primary_alteration,secondary_degree,secondary_alteration,quality,inversion
    ... bar,na,na,na,na,na,na
    ... note,I,_,I,_,M,
    ... note,VII,#,V,_,d,43
    ... bar,,,,
    ... note,V,_,VII,b,M,6
    ... note,VI,#,II,b,m,64
    ... note,VI,b,I,_,M,
    ... '''
    ...     )
    ... )
    >>> split_degrees_to_single_degree(df)["degree"]
    0         na
    1          I
    2     #VII/V
    3         na
    4     V/bVII
    5    #VI/bII
    6        bVI
    Name: degree, dtype: object
    >>> split_degrees_to_single_degree(
    ...     df, inversion_col="inversion", quality_col="quality", output_col="rn"
    ... )["rn"]
    0            na
    1            IM
    2     #VIId43/V
    3            na
    4      VM6/bVII
    5    #VIm64/bII
    6          bVIM
    Name: rn, dtype: object
    """
    if not inplace:
        df = df.copy()

    df[output_col] = (
        df[primary_alteration_col]
        + df[primary_degree_col]
        + (df[quality_col].fillna("") if quality_col is not None else "")
        + (df[inversion_col].fillna("") if inversion_col is not None else "")
        + "/"
        + df[secondary_alteration_col]
        + df[secondary_degree_col]
    )
    df[output_col] = df[output_col].str.replace(null_alteration_char, "")

    # Remove "/I" from the end of Roman numerals
    df[output_col] = df[output_col].str.replace(r"/I$", "", regex=True)

    if "type" in df.columns:
        df.loc[df["type"] != "note", output_col] = null_chord_token

    return df


def single_degree_to_split_degrees(
    df: pd.DataFrame,
    degree_col: str = "degree",
    primary_degree_col: str = "primary_degree",
    primary_alteration_col: str = "primary_alteration",
    secondary_degree_col: str = "secondary_degree",
    secondary_alteration_col: str = "secondary_alteration",
    null_alteration_char: str = "_",
    null_chord_token: str = "na",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... degree
    ... I
    ... IV
    ... V
    ... I
    ... '''
    ...     )
    ... )
    >>> single_degree_to_split_degrees(df)
      degree primary_degree primary_alteration secondary_degree secondary_alteration
    0      I              I                  _                I                    _
    1     IV             IV                  _                I                    _
    2      V              V                  _                I                    _
    3      I              I                  _                I                    _

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... degree
    ... I
    ... VII/V
    ... V/bVII
    ... #VI/bII
    ... bVI
    ... na
    ... '''
    ...     )
    ... )
    >>> single_degree_to_split_degrees(df)
        degree primary_degree primary_alteration secondary_degree secondary_alteration
    0        I              I                  _                I                    _
    1    VII/V            VII                  _                V                    _
    2   V/bVII              V                  _              VII                    b
    3  #VI/bII             VI                  #               II                    b
    4      bVI             VI                  b                I                    _
    5       na             na                 na               na                   na
    """
    if not inplace:
        df = df.copy()

    splits = df[degree_col].str.split("/", n=1, expand=True)

    null_mask = (df[degree_col] == null_chord_token) | (df[degree_col].isna())

    primary = (
        splits[0]
        .str.extract(r"([b#]*)(.*)")
        .rename(columns={0: primary_alteration_col, 1: primary_degree_col})
    )
    primary[primary_alteration_col] = (
        primary[primary_alteration_col]
        .fillna(null_alteration_char)
        .replace("", null_alteration_char)
    )
    primary.loc[null_mask, :] = null_chord_token
    df[primary_degree_col] = primary[primary_degree_col]
    df[primary_alteration_col] = primary[primary_alteration_col]

    if splits.shape[1] == 1:
        # There are no secondary degrees
        df[secondary_degree_col] = "I"
        df[secondary_alteration_col] = null_alteration_char

    else:
        secondary = (
            splits[1]
            .str.extract(r"([b#]*)(.*)")
            .rename(columns={0: secondary_alteration_col, 1: secondary_degree_col})
        )

        secondary[secondary_alteration_col] = (
            secondary[secondary_alteration_col]
            .fillna(null_alteration_char)
            .replace("", null_alteration_char)
        )
        secondary[secondary_degree_col] = secondary[secondary_degree_col].fillna("I")

        secondary.loc[null_mask, :] = null_chord_token

        df[secondary_degree_col] = secondary[secondary_degree_col]
        df[secondary_alteration_col] = secondary[secondary_alteration_col]

    return df


def drop_harmony_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If, for example, we are re-labeling a score, we want to drop all columns that may
    have chord annotations to be sure they don't leak through.
    """
    patterns = [
        "key",
        "degree",
        "mode",
        "rn",
        "chord",
        "harmony",
        "bass",
        "alteration",
        "inversion",
        "quality",
        "root",
    ]
    harmony_cols = [
        c for c in df.columns if any(pattern in c.lower() for pattern in patterns)
    ]
    # print(f"Removing columns: {harmony_cols}")
    df = df.drop(columns=harmony_cols)
    # print(f"Remaining columns: {df.columns}")
    return df


def extract_chord_df_from_music_df(
    music_df: pd.DataFrame,
    null_chord_token: str = "na",
    columns: Iterable[str] = ("key", "onset", "degree", "quality", "inversion"),
    release_col: str = "release",
) -> pd.DataFrame:
    """

    >>> music_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,key,onset,release,degree,quality,inversion
    ... bar,,,0.0,4.0,,,
    ... note,60,C,0.0,1.0,I,M,0.0
    ... note,64,C,1.0,2.0,I,M,0.0
    ... note,62,C,2.0,3.0,V,M,1.0
    ... note,67,C,3.0,4.0,V,M,0.0
    ... note,66,G,4.0,6.0,V,M,0.0
    ... '''
    ...     )
    ... )
    >>> extract_chord_df_from_music_df(music_df)
      key  onset degree quality  inversion  release
    0   C    0.0      I       M        0.0      2.0
    1   C    2.0      V       M        1.0      3.0
    2   C    3.0      V       M        0.0      4.0
    3   G    4.0      V       M        0.0      6.0
    """
    columns = list(columns)
    assert all(col in music_df.columns for col in columns), (
        f"music_df must have the following columns: {columns}"
    )

    chord_change_masks = [
        music_df[col] != music_df[col].shift(1) for col in columns if col != "onset"
    ]
    chord_change_mask = np.logical_or.reduce(chord_change_masks)

    chord_df = music_df.loc[chord_change_mask, columns].copy()
    chord_df = chord_df.loc[
        (chord_df[columns[0]] != null_chord_token) & (~chord_df[columns[0]].isna())
    ]
    chord_df = chord_df.reset_index(drop=True)

    if release_col in music_df.columns:
        chord_df[release_col] = chord_df["onset"].shift(-1)
        # We use .max() rather than .iloc[-1] because the last item in music_df may
        # not have a release time, e.g. if it is a bar.
        chord_df.loc[len(chord_df) - 1, release_col] = music_df[release_col].max()

    return chord_df


def label_music_df_with_chord_df(
    music_df: pd.DataFrame,
    chord_df: pd.DataFrame,
    columns_to_add: Iterable[str] = ("key", "degree", "quality", "inversion"),
    null_chord_token: str = "na",
) -> pd.DataFrame:
    """
    >>> music_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,1.0
    ... note,64,1.0,2.0
    ... note,62,2.0,3.0
    ... note,67,3.0,4.0
    ... bar,,4.0,8.0
    ... note,66,4.0,6.0
    ... note,67,6.0,8.0
    ... '''
    ...     )
    ... )
    >>> chord_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,key,degree,quality,inversion
    ... 0.0,C,I,M,0.0
    ... 3.0,C,V,M,1.0
    ... 5.0,G,V,M,0.0
    ... 7.0,G,I,M,0.0
    ... '''
    ...     )
    ... )
    >>> label_music_df_with_chord_df(music_df, chord_df)
       type  pitch  onset  release key degree quality  inversion
    0   bar    NaN    0.0      4.0  na     na      na        NaN
    1  note   60.0    0.0      1.0   C      I       M        0.0
    2  note   64.0    1.0      2.0   C      I       M        0.0
    3  note   62.0    2.0      3.0   C      I       M        0.0
    4  note   67.0    3.0      4.0   C      V       M        1.0
    5   bar    NaN    4.0      8.0  na     na      na        NaN
    6  note   66.0    4.0      6.0   C      V       M        1.0
    7  note   67.0    6.0      8.0   G      V       M        0.0
    """
    out = pd.merge_asof(
        music_df.drop(columns=[c for c in columns_to_add if c in music_df.columns]),
        chord_df[["onset"] + list(columns_to_add)],
        on="onset",
        direction="backward",
    )
    nonnote_mask = out["type"] != "note"
    for col in columns_to_add:
        if out[col].dtype == "object":
            out.loc[nonnote_mask, col] = null_chord_token
        else:
            out.loc[nonnote_mask, col] = float("nan")

    return out
