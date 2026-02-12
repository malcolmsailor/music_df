import io  # noqa: F401

import numpy as np
import pandas as pd

from music_df.harmony.chords import (
    CacheDict,
    handle_nested_secondary_rns,
    spelled_pitch_to_rn,
    tonicization_to_key,
)


def assert_range_index(df: pd.DataFrame):
    try:
        assert isinstance(df.index, pd.RangeIndex)
    except AssertionError:
        assert (df.index == range(len(df))).all()
        return

    assert df.index.start == 0
    assert df.index.stop == len(df)
    assert df.index.step == 1


def expand_tonicizations(df: pd.DataFrame, quality_col: str | None = None):
    """
    If we have a sequence of annotations like "vi V/vi vi ii/vi V/vi vi", for the
    purposes of determining the length of tonicizations, we want to treat it as
    "i/vi V/vi i/vi ii/vi V/vi i/vi".

    Thus this function looks for tonicizations that are preceded/followed by the
    tonicized degree and then "expands" them by treating that degree as "i/...".

    We make a special case for dominant chords (i.e., chords that have degree "V/I"
    or "V/i"). "V/V V I" is such an ubiquitous pattern that it seems like "expanding"
    it isn't appropriate (in particular since V is very often a V7 chord and thus not
    a plausible tonic). Therefore, we only expand dominant chords that are both
    preceded and followed by tonicization (e.g., "V/V V V/V" but not "V/V V I").

    `replace_spurious_tonics` is a quasi-inverse of this function.

    If `quality_col` is provided, we only expand tonicized chords that are not
    dominant sevenths, augmented sixths, or diminished or augmented triads
    (i.e., quality doesn't contain "Mm7", "aug6", "o", or "+").

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,key
    ... vi,i,C
    ... V,vi,C
    ... vi,i,C
    ... ii,vi,C
    ... V,vi,C
    ... vi,i,C
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df)
      primary_degree secondary_degree key
    0              i               vi   C
    1              V               vi   C
    2              i               vi   C
    3             ii               vi   C
    4              V               vi   C
    5              i               vi   C

    Across key-change
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,key
    ... vi,i,C
    ... V,vi,G
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df)
      primary_degree secondary_degree key
    0             vi                i   C
    1              V               vi   G
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,key
    ... IV,ii,Ab
    ... ii,I,c
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df)
      primary_degree secondary_degree key
    0             IV               ii  Ab
    1             ii                I   c

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,key
    ... vi,i,C
    ... V,vi,C
    ... vi,i,C
    ... ii,VI,C
    ... V,VI,C
    ... VI,i,C
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df)
      primary_degree secondary_degree key
    0              i               vi   C
    1              V               vi   C
    2              i               vi   C
    3             ii               VI   C
    4              V               VI   C
    5              I               VI   C

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,key
    ... vi,i,C
    ... V,VI,C
    ... vi,i,C
    ... ii,VI,C
    ... V,VI,C
    ... VI,i,C
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df)
      primary_degree secondary_degree key
    0             vi                i   C
    1              V               VI   C
    2             vi                i   C
    3             ii               VI   C
    4              V               VI   C
    5              I               VI   C

    Dominant chord behavior
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,key
    ... V,V,Ab
    ... V,I,Ab
    ... V,V,Ab
    ... '''
    ...     )
    ... )

    When V is preceded and followed by tonicization, it is expanded
    >>> expand_tonicizations(df.copy())
      primary_degree secondary_degree key
    0              V                V  Ab
    1              I                V  Ab
    2              V                V  Ab

    When V is only preceded or only followed by tonicization, it is not expanded
    >>> expand_tonicizations(df.iloc[1:].copy())
      primary_degree secondary_degree key
    1              V                I  Ab
    2              V                V  Ab
    >>> expand_tonicizations(df.iloc[:-1].copy())
      primary_degree secondary_degree key
    0              V                V  Ab
    1              V                I  Ab

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,key,quality
    ... V,V,Ab,M
    ... V,I,Ab,Mm7
    ... V,V,Ab,M
    ... V,I,Ab,aug6
    ... V,V,Ab,M
    ... V,I,Ab,o
    ... V,V,Ab,M
    ... V,I,Ab,+
    ... V,V,Ab,M
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df, quality_col="quality")
      primary_degree secondary_degree key quality
    0              V                V  Ab       M
    1              V                I  Ab     Mm7
    2              V                V  Ab       M
    3              V                I  Ab    aug6
    4              V                V  Ab       M
    5              V                I  Ab       o
    6              V                V  Ab       M
    7              V                I  Ab       +
    8              V                V  Ab       M
    """

    non_dominant_mask = (
        (df["primary_degree"] != "V")
        & (df["secondary_degree"].isin(["i", "I"]))
        & (
            (
                (df["primary_degree"] == df["secondary_degree"].shift(-1))
                & (df["key"] == df["key"].shift(-1))
            )
            | (
                (df["primary_degree"] == df["secondary_degree"].shift(1))
                & (df["key"] == df["key"].shift(1))
            )
        )
    )

    dominant_mask = (
        (df["primary_degree"] == "V")
        & (df["secondary_degree"].isin(["i", "I"]))
        & (
            (df["primary_degree"] == df["secondary_degree"].shift(-1))
            & (df["key"] == df["key"].shift(-1))
            & (df["primary_degree"] == df["secondary_degree"].shift(1))
            & (df["key"] == df["key"].shift(1))
        )
    )

    if quality_col is not None:
        non_dominant_mask &= ~df[quality_col].str.contains("Mm7|aug6|o|\\+")
        dominant_mask &= ~df[quality_col].str.contains("Mm7|aug6|o|\\+")

    def _apply_mask(mask):
        df.loc[mask, "secondary_degree"] = df.loc[mask, "primary_degree"]
        lower_case_indices = df.loc[mask].index[
            df.loc[mask, "primary_degree"].str.slice(start=-1).str.islower()
        ]

        if "secondary_mode" in df.columns:
            upper_case_indices = df.loc[mask].index[
                df.loc[mask, "primary_degree"].str.slice(start=-1).str.isupper()
            ]
            df.loc[lower_case_indices, "secondary_mode"] = "m"
            df.loc[upper_case_indices, "secondary_mode"] = "M"

        df.loc[mask, "primary_degree"] = "I"
        df.loc[lower_case_indices, "primary_degree"] = "i"

    _apply_mask(non_dominant_mask)
    _apply_mask(dominant_mask)

    return df


def split_degree_into_primary_and_secondary(
    chord_df: pd.DataFrame,
    degree_col: str = "degree",
    primary_degree_col: str = "primary_degree",
    secondary_degree_col: str = "secondary_degree",
    secondary_mode_col: str = "secondary_mode",
    inplace: bool = True,
):
    """
    >>> df = pd.DataFrame({"degree": ["I", "V/VI", "V/VIM", "V/vim", "IV"]})
    >>> split_degree_into_primary_and_secondary(df)  # doctest: +NORMALIZE_WHITESPACE
      degree primary_degree secondary_degree secondary_mode
    0      I              I                I              _
    1   V/VI              V               VI              _
    2  V/VIM              V               VI              M
    3  V/vim              V               vi              m
    4     IV             IV                I              _
    """
    if not inplace:
        chord_df = chord_df.copy()

    split_result = chord_df[degree_col].str.split("/", n=1, expand=True)
    if split_result.shape[1] == 2:
        chord_df[primary_degree_col] = split_result[0]
        # Extract secondary degree and optional mode suffix
        secondary_parts = split_result[1].str.extract(r"^([b#]*[IViv]+)([mM]?)$")
        chord_df[secondary_degree_col] = secondary_parts[0].fillna("I")
        chord_df[secondary_mode_col] = secondary_parts[1].fillna("_").replace("", "_")
    else:
        chord_df[primary_degree_col] = split_result[0]
        chord_df[secondary_degree_col] = "I"
        chord_df[secondary_mode_col] = "_"

    return chord_df


def _reconstruct_degree_column(chord_df: pd.DataFrame) -> pd.Series:
    secondary_with_slash = "/" + chord_df["secondary_degree"]
    if "secondary_mode" in chord_df.columns:
        mode_suffix = chord_df["secondary_mode"].fillna("_").replace("_", "")
        secondary_with_slash = secondary_with_slash + mode_suffix
    secondary_with_slash = secondary_with_slash.replace("/I", "")
    return chord_df["primary_degree"] + secondary_with_slash


def remove_long_tonicizations(
    chord_df: pd.DataFrame,
    inplace: bool = False,
    max_tonicization_duration: float | None = None,
    min_removal_duration: float | None = None,
    max_tonicization_num_chords: int | None = None,
    min_removal_num_chords: int | None = None,
    tonicization_cache: CacheDict[tuple[str, str, str | None], str] | None = None,
    case_matters: bool = False,
    simplify_enharmonics: bool = True,
) -> pd.DataFrame:
    """
    Remove long tonicizations from a chord dataframe.

    Note that we assume that all tonicizations are "normalized" in the sense that
    they only use a single slash and secondary RN. (E.g., we don't have RNs like
    "V/V/V".)

    Note that we ALSO assume that all harmonies are de-repeated, i.e., that identical
    harmonies (e.g., "I6") aren't repeated more than once, which can break some of
    the behavior here.

    Args:
        chord_df: A dataframe containing either of the following sets of columns:
            - "onset", "primary_degree", "secondary_degree", and "key"
            - "onset", "degree", and "key"
            In the former case, if the dataframe has an "degree" column, it will be
            overwritten. If a "quality" column is provided, it is used when
            determining whether to expand tonicizations.
        inplace: Whether to modify the dataframe in place.
        max_tonicization_duration: The maximum duration of a tonicization. At least one
            of max_tonicization_duration or max_tonicization_num_chords must be provided.
        min_removal_duration: If provided, a tonicization must have at least this
            duration to be removed. This argument must be <= max_tonicization_duration
            if both are provided.
        max_tonicization_num_chords: The maximum number of chords in a tonicization.
            At least one of max_tonicization_duration or max_tonicization_num_chords
            must be provided. Note that repetitions of the same chord count as multiple
            chords. (If you wish for a different behavior, you should remove chord
            repetitions in advance.)
        min_removal_num_chords: If provided, a tonicization must have at least this
            number of consecutive chords to be removed. Concerning chord repetitions,
            see the note above about max_tonicization_num_chords. This argument must be
            <= max_tonicization_num_chords if both are provided.
        tonicization_cache: A cache for tonicizations to save compute if running
            this function many times.
        case_matters: Whether to consider case when determining the key of the
            tonicization. Ignored if tonicization_cache is provided.
        simplify_enharmonics: Whether to simplify enharmonic spellings of the key of
            the tonicization. Ignored if tonicization_cache is provided.



    >>> no_tonicizations = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(no_tonicizations, max_tonicization_num_chords=1)
       onset degree key
    0    0.0      I   C

    >>> consecutive_tonicizations = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,Db
    ... 1.0,V/#IV,Db
    ... 2.0,IV/#IV,Db
    ... 3.0,I/#IV,Db
    ... 4.0,V/bVI,Db
    ... 5.0,I/bVI,Db
    ... 6.0,I,Db
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(
    ...     consecutive_tonicizations, max_tonicization_num_chords=1
    ... )
       onset degree key
    0    0.0      I  Db
    1    1.0      V   G
    2    2.0     IV   G
    3    3.0      I   G
    4    4.0      V   A
    5    5.0      I   A
    6    6.0      I  Db


    >>> tonicization_with_key_change = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,a
    ... 1.0,IV/III,a
    ... 2.0,V/III,a
    ... 3.0,V/III,d
    ... 4.0,IV/III,d
    ... 5.0,I,d
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(
    ...     tonicization_with_key_change, max_tonicization_num_chords=1
    ... )
       onset degree key
    0    0.0      I   a
    1    1.0     IV   C
    2    2.0      V   C
    3    3.0      V   F
    4    4.0     IV   F
    5    5.0      I   d

    >>> tonicization_and_key_change = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,bb
    ... 1.0,ii/iv,bb
    ... 2.0,V/iv,bb
    ... 3.0,ii/III,eb
    ... 4.0,V/III,eb
    ... 5.0,I,eb
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(
    ...     tonicization_and_key_change, max_tonicization_num_chords=1
    ... )
       onset degree key
    0    0.0      I  bb
    1    1.0     ii  eb
    2    2.0      V  eb
    3    3.0     ii  Gb
    4    4.0      V  Gb
    5    5.0      I  eb

    >>> tonicization_at_beginning_and_end = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,V/V,C
    ... 1.0,viio/V,C
    ... 2.0,I,C
    ... 3.0,iv/ii,C
    ... 4.0,V/ii,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(
    ...     tonicization_at_beginning_and_end, max_tonicization_num_chords=1
    ... )
       onset degree key
    0    0.0      V   G
    1    1.0   viio   G
    2    2.0      I   C
    3    3.0     iv   d
    4    4.0      V   d

    >>> single_chord_tonicization = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,release,degree,key
    ... 0.0,1.0,V/V,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(
    ...     single_chord_tonicization, max_tonicization_num_chords=0
    ... )
       onset  release degree key
    0    0.0      1.0      V   G

    Note that tonicizations are first "expanded", and then "spurious" tonicizations are
    removed. This means that we treat chords that are adjacent to their tonicizations
    (e.g., "V" next to "V/V") as belonging to the tonicized region, but before
    returning, we remove any "spurious" tonicizations (e.g., replacing "I/V" with "V").
    >>> chord_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... 0.5,V,C
    ... 1,V/V,C
    ... 2,I/V,C
    ... 3,V/V,C
    ... 4,I/V,C
    ... 5,V/V,C
    ... 6,I/V,C
    ... 7,V/V,C
    ... 8,V,C
    ... 9,I,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(chord_df, max_tonicization_num_chords=9)
        onset degree key
    0     0.0      I   C
    1     0.5      V   C
    2     1.0    V/V   C
    3     2.0      V   C
    4     3.0    V/V   C
    5     4.0      V   C
    6     5.0    V/V   C
    7     6.0      V   C
    8     7.0    V/V   C
    9     8.0      V   C
    10    9.0      I   C
    >>> remove_long_tonicizations(chord_df, max_tonicization_num_chords=6)
        onset degree key
    0     0.0      I   C
    1     0.5      V   C
    2     1.0      V   G
    3     2.0      I   G
    4     3.0      V   G
    5     4.0      I   G
    6     5.0      V   G
    7     6.0      I   G
    8     7.0      V   G
    9     8.0      V   C
    10    9.0      I   C
    >>> remove_long_tonicizations(chord_df, max_tonicization_duration=9.0)
        onset degree key
    0     0.0      I   C
    1     0.5      V   C
    2     1.0    V/V   C
    3     2.0      V   C
    4     3.0    V/V   C
    5     4.0      V   C
    6     5.0    V/V   C
    7     6.0      V   C
    8     7.0    V/V   C
    9     8.0      V   C
    10    9.0      I   C
    >>> remove_long_tonicizations(chord_df, max_tonicization_duration=6.9)
        onset degree key
    0     0.0      I   C
    1     0.5      V   C
    2     1.0      V   G
    3     2.0      I   G
    4     3.0      V   G
    5     4.0      I   G
    6     5.0      V   G
    7     6.0      I   G
    8     7.0      V   G
    9     8.0      V   C
    10    9.0      I   C

    >>> long_single_chord_tonicization = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... 1.0,V/V,C
    ... 9.0,I,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(
    ...     long_single_chord_tonicization, max_tonicization_duration=7.9
    ... )
       onset degree key
    0    0.0      I   C
    1    1.0      V   G
    2    9.0      I   C
    >>> remove_long_tonicizations(
    ...     long_single_chord_tonicization,
    ...     max_tonicization_duration=7.9,
    ...     min_removal_num_chords=3,
    ... )
       onset degree key
    0    0.0      I   C
    1    1.0    V/V   C
    2    9.0      I   C

    >>> multiple_brief_chords = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... 1.0,ii/V,C
    ... 1.125,V/V,C
    ... 1.25,ii/V,C
    ... 1.375,V/V,C
    ... 1.5,V,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(multiple_brief_chords, max_tonicization_num_chords=1)
       onset degree key
    0  0.000      I   C
    1  1.000     ii   G
    2  1.125      V   G
    3  1.250     ii   G
    4  1.375      V   G
    5  1.500      V   C
    >>> remove_long_tonicizations(
    ...     multiple_brief_chords,
    ...     max_tonicization_num_chords=1,
    ...     min_removal_duration=1,
    ... )
       onset degree key
    0  0.000      I   C
    1  1.000   ii/V   C
    2  1.125    V/V   C
    3  1.250   ii/V   C
    4  1.375    V/V   C
    5  1.500      V   C

    >>> consecutive_chords = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,V/V,C
    ... 1.0,V/V,C
    ... 2.0,V/V,C
    ... 3.0,V/V,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(consecutive_chords, max_tonicization_num_chords=3)
       onset degree key
    0    0.0      V   G
    1    1.0      V   G
    2    2.0      V   G
    3    3.0      V   G

    173.0,173.5,7a2,II,_,I,_,0.0,F,m
    173.5,174.0,580,IV,_,V,_,0.0,F,m
    174.0,174.5,5b2,VII,#,V,_,2.0,F,o
    174.5,175.0,370,V,_,I,_,1.0,F,m
    175.0,175.5,370,V,_,I,_,1.0,F,m
    175.5,176.0,b27,V,_,V,_,1.0,F,M
    176.0,176.5,b27,V,_,V,_,1.0,F,M
    176.5,177.0,037,V,_,I,_,0.0,F,m
    177.0,177.5,037,V,_,I,_,0.0,F,m
    177.5,178.0,a25,IV,_,I,_,0.0,F,M
    178.0,178.5,a47,VII,_,I,_,2.0,F,o
    178.5,179.5,905,I,_,I,_,1.0,F,M
    179.5,180.0,590,I,_,I,_,0.0,F,M
    180.0,180.5,5903,V,_,IV,_,0.0,F,Mm7


    """
    assert (
        max_tonicization_duration is not None or max_tonicization_num_chords is not None
    )

    if min_removal_num_chords is not None and max_tonicization_num_chords is not None:
        assert min_removal_num_chords <= max_tonicization_num_chords

    if min_removal_duration is not None and max_tonicization_duration is not None:
        assert min_removal_duration <= max_tonicization_duration

    if not inplace:
        chord_df = chord_df.copy()

    assert_range_index(chord_df)

    split_columns = ["primary_degree", "secondary_degree", "key", "onset"]
    joined_columns = [
        "onset",
        "degree",
        "key",
    ]
    had_secondary_mode = "secondary_mode" in chord_df.columns
    if all(k in chord_df.columns for k in split_columns):
        had_split_columns = True
    elif all(k in chord_df.columns for k in joined_columns):
        had_split_columns = False
        chord_df = split_degree_into_primary_and_secondary(chord_df)
    else:
        raise ValueError(
            f"chord_df must have columns {split_columns} or {joined_columns}"
        )

    # We begin by filling the key column so that if we change the key during a
    # tonicization, it will revert to the original key after the tonicized passage.

    chord_df["key"] = chord_df["key"].ffill()

    # We fill the secondary_degree column with "I"
    chord_df["secondary_degree"] = chord_df["secondary_degree"].fillna("I")

    chord_df = expand_tonicizations(
        chord_df, quality_col="quality" if "quality" in chord_df.columns else None
    )

    tonicization_changes = (
        chord_df["secondary_degree"] != chord_df["secondary_degree"].shift(1)
    ) | (chord_df["key"] != chord_df["key"].shift(1))

    indices = tonicization_changes.index[tonicization_changes].tolist()
    indices.append(len(chord_df))

    def get_tonicized_key(
        secondary_degree: str, key: str, secondary_mode: str | None = None
    ):
        if tonicization_cache is None:
            return tonicization_to_key(
                secondary_degree,
                key,
                case_matters,
                simplify_enharmonics,
                secondary_mode,
            )
        else:
            return tonicization_cache[(secondary_degree, key, secondary_mode)]

    def remove_tonicization(tonicization: str, start_i: int, end_i: int):
        secondary_mode = None
        if "secondary_mode" in chord_df.columns:
            sm = chord_df.iloc[start_i]["secondary_mode"]
            if sm in ("M", "m"):
                secondary_mode = sm
        new_key = get_tonicized_key(
            tonicization, chord_df.iloc[start_i]["key"], secondary_mode
        )
        chord_df.loc[start_i : end_i - 1, "key"] = new_key
        chord_df.loc[start_i : end_i - 1, "secondary_degree"] = "I"
        if "secondary_mode" in chord_df.columns:
            chord_df.loc[start_i : end_i - 1, "secondary_mode"] = "_"

    for start_i, end_i in zip(indices[:-1], indices[1:]):
        assert chord_df.iloc[start_i:end_i]["secondary_degree"].nunique() == 1
        assert chord_df.iloc[start_i:end_i]["key"].nunique() == 1
        tonicization = chord_df.iloc[start_i]["secondary_degree"]
        if tonicization == "I":
            continue

        if end_i == len(chord_df):
            # Length of tonicization is undefined
            tonicization_duration = float("inf")
        else:
            tonicization_onset = chord_df.iloc[start_i]["onset"]
            tonicization_release = chord_df.iloc[end_i]["onset"]
            tonicization_duration = tonicization_release - tonicization_onset

        if (
            max_tonicization_num_chords is not None
            and end_i - start_i > max_tonicization_num_chords
        ):
            if (
                min_removal_duration is None
                or tonicization_duration >= min_removal_duration
            ):
                remove_tonicization(tonicization, start_i, end_i)
                continue

        if (
            max_tonicization_duration is not None
            and tonicization_duration > max_tonicization_duration
        ):
            if (
                min_removal_num_chords is None
                or end_i - start_i >= min_removal_num_chords
            ):
                remove_tonicization(tonicization, start_i, end_i)

    chord_df = replace_spurious_tonics(chord_df)

    # Drop secondary_mode before reconstruction if it wasn't in the original input,
    # so mode suffixes don't leak into the degree column.
    if not had_secondary_mode and "secondary_mode" in chord_df.columns:
        chord_df = chord_df.drop(columns=["secondary_mode"])

    chord_df["degree"] = _reconstruct_degree_column(chord_df)

    if not had_split_columns:
        chord_df = chord_df.drop(columns=["primary_degree", "secondary_degree"])

    return chord_df


def replace_spurious_tonics(chord_df: pd.DataFrame, inplace: bool = False):
    """
    Replace chords like "I/V" with chords like "V".

    Used by remove_short_modulations().

    >>> spurious_tonic = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,primary_degree,secondary_degree,key
    ... 0.0,I,I,C
    ... 1.0,I,V,C
    ... '''
    ...     )
    ... )
    >>> replace_spurious_tonics(spurious_tonic)
       onset primary_degree secondary_degree key
    0    0.0              I                I   C
    1    1.0              V                I   C
    """
    assert_range_index(chord_df)

    split_columns = ["primary_degree", "secondary_degree", "key", "onset"]
    assert all(k in chord_df.columns for k in split_columns)

    if not inplace:
        chord_df = chord_df.copy()

    chord_df["secondary_degree"] = chord_df["secondary_degree"].fillna("I")

    spurious_tonic_mask = (chord_df["primary_degree"] == "I") & (
        chord_df["secondary_degree"] != "I"
    )
    chord_df.loc[spurious_tonic_mask, "primary_degree"] = chord_df.loc[
        spurious_tonic_mask, "secondary_degree"
    ]
    chord_df.loc[spurious_tonic_mask, "secondary_degree"] = "I"
    if "secondary_mode" in chord_df.columns:
        chord_df.loc[spurious_tonic_mask, "secondary_mode"] = "_"

    return chord_df


def remove_short_modulations(
    chord_df: pd.DataFrame,
    inplace: bool = False,
    min_modulation_duration: float | None = None,
    max_removal_duration: float | None = None,
    min_modulation_num_chords: int | None = None,
    max_removal_num_chords: int | None = None,
    spelled_pitch_to_rn_cache: CacheDict[tuple[str, str], str] | None = None,
    handle_nested_secondary_rns_cache: CacheDict[tuple[str, str], str] | None = None,
):
    """
    Replace short modulations with the equivalent tonicization.

    Note that we only replace modulations that are preceded and followed by the same
    key.





    >>> modulation_with_tonicization = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,b
    ... 1.0,V/V,G
    ... 2.0,V,G
    ... 3.0,VI,b
    ... '''
    ...     )
    ... )
    >>> remove_short_modulations(
    ...     modulation_with_tonicization, min_modulation_num_chords=3
    ... )
       onset degree key
    0    0.0      I   b
    1    1.0  V/III   b
    2    2.0   V/VI   b
    3    3.0     VI   b


    >>> one_chord_modulation = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... 1.0,I,G
    ... 2.0,I,C
    ... '''
    ...     )
    ... )
    >>> remove_short_modulations(one_chord_modulation, min_modulation_num_chords=2)
       onset degree key
    0    0.0      I   C
    1    1.0      V   C
    2    2.0      I   C

    >>> brief_modulation = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,Ab
    ... 1.0,I,C
    ... 1.25,viio6,C
    ... 1.5,I,C
    ... 2.0,I,Ab
    ... '''
    ...     )
    ... )
    >>> remove_short_modulations(brief_modulation, min_modulation_duration=2.0)
       onset     degree key
    0   0.00          I  Ab
    1   1.00        III  Ab
    2   1.25  viio6/III  Ab
    3   1.50        III  Ab
    4   2.00          I  Ab

    >>> modulation_at_end = pd.read_csv(  # Should remain unchanged
    ...     # Of course, you could also interpret this as a modulation at the beginning
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,F#
    ... 1.0,V,G
    ... 2.0,I,G
    ... '''
    ...     )
    ... )
    >>> remove_short_modulations(modulation_at_end, min_modulation_num_chords=3)
       onset degree key
    0    0.0      I  F#
    1    1.0      V   G
    2    2.0      I   G

    """
    assert (
        min_modulation_duration is not None or min_modulation_num_chords is not None
    ), (
        "At least one of min_modulation_duration or min_modulation_num_chords must be provided"
    )

    if min_modulation_num_chords is not None and max_removal_num_chords is not None:
        assert min_modulation_num_chords <= max_removal_num_chords

    if min_modulation_duration is not None and max_removal_duration is not None:
        assert min_modulation_duration <= max_removal_duration

    if not inplace:
        chord_df = chord_df.copy()

    assert_range_index(chord_df)

    split_columns = ["primary_degree", "secondary_degree", "key", "onset"]
    joined_columns = [
        "onset",
        "degree",
        "key",
    ]
    had_secondary_mode = "secondary_mode" in chord_df.columns
    if all(k in chord_df.columns for k in split_columns):
        had_split_columns = True
    elif all(k in chord_df.columns for k in joined_columns):
        had_split_columns = False
        chord_df = split_degree_into_primary_and_secondary(chord_df)
    else:
        raise ValueError(
            f"chord_df must have columns {split_columns} or {joined_columns}"
        )

    chord_df["key"] = chord_df["key"].ffill()

    key_changes = chord_df["key"] != chord_df["key"].shift(1)
    indices = key_changes.index[key_changes].tolist()
    indices.append(len(chord_df))

    def get_secondary_rn(inner_key: str, outer_key: str):
        inner_key_mode = "M" if inner_key[0].isupper() else "m"

        if spelled_pitch_to_rn_cache is None:
            return spelled_pitch_to_rn(inner_key, outer_key, inner_key_mode)
        else:
            return spelled_pitch_to_rn_cache[(inner_key, outer_key, inner_key_mode)]

    def nested_handler(row: pd.Series, secondary_degree: str):
        if handle_nested_secondary_rns_cache is None:
            return handle_nested_secondary_rns(
                row["secondary_degree"] + "/" + secondary_degree, row["key"]
            )
        else:
            return handle_nested_secondary_rns_cache[
                (row["secondary_degree"] + "/" + secondary_degree, row["key"])
            ]

    def remove_modulation_if_possible(start_i: int, end_i: int):
        if start_i == 0:
            return
        if end_i == len(chord_df):
            return
        if chord_df.loc[start_i - 1, "key"] != chord_df.loc[end_i, "key"]:
            return
        outer_key = chord_df.loc[start_i - 1, "key"]
        inner_key = chord_df.loc[start_i, "key"]
        inner_key_mode = "M" if inner_key[0].isupper() else "m"
        secondary_degree = get_secondary_rn(inner_key, outer_key)

        # Handle nested secondary RNs
        nested_secondary_mask = (
            chord_df.loc[start_i : end_i - 1, "secondary_degree"] != "I"
        )
        nested_secondary_index = nested_secondary_mask.index[nested_secondary_mask]
        chord_df.loc[nested_secondary_index, "secondary_degree"] = chord_df.loc[
            nested_secondary_index
        ].apply(lambda row: nested_handler(row, secondary_degree), axis=1)

        if "secondary_mode" in chord_df.columns:
            # For nested secondaries, derive mode from case of the collapsed result
            for idx in nested_secondary_index:
                collapsed = chord_df.loc[idx, "secondary_degree"]
                collapsed_mode = "M" if collapsed.lstrip("#b")[-1].isupper() else "m"
                chord_df.loc[idx, "secondary_mode"] = collapsed_mode

        # Handle simple secondary RNs
        simple_secondary_index = nested_secondary_mask.index[~nested_secondary_mask]
        chord_df.loc[simple_secondary_index, "secondary_degree"] = secondary_degree

        if "secondary_mode" in chord_df.columns:
            chord_df.loc[simple_secondary_index, "secondary_mode"] = inner_key_mode

        chord_df.loc[start_i : end_i - 1, "key"] = outer_key

    for start_i, end_i in zip(indices[:-1], indices[1:]):
        assert chord_df.iloc[start_i:end_i]["key"].nunique() == 1

        if end_i == len(chord_df):
            # Length of modulation is undefined
            modulation_duration = float("inf")
        else:
            modulation_onset = chord_df.iloc[start_i]["onset"]
            modulation_release = chord_df.iloc[end_i]["onset"]
            modulation_duration = modulation_release - modulation_onset

        if (
            min_modulation_num_chords is not None
            and end_i - start_i < min_modulation_num_chords
        ):
            if (
                max_removal_duration is None
                or modulation_duration <= max_removal_duration
            ):
                remove_modulation_if_possible(start_i, end_i)
                continue

        if (
            min_modulation_duration is not None
            and modulation_duration < min_modulation_duration
        ):
            if (
                max_removal_num_chords is None
                or end_i - start_i <= max_removal_num_chords
            ):
                remove_modulation_if_possible(start_i, end_i)

    replace_spurious_tonics(chord_df, inplace=True)

    if not had_secondary_mode and "secondary_mode" in chord_df.columns:
        chord_df = chord_df.drop(columns=["secondary_mode"])

    chord_df["degree"] = _reconstruct_degree_column(chord_df)

    if not had_split_columns:
        chord_df = chord_df.drop(columns=["primary_degree", "secondary_degree"])

    return chord_df


def tonicization_census(
    chord_df: pd.DataFrame,
    degree_col: str = "degree",
    primary_degree_col: str = "primary_degree",
    secondary_degree_col: str = "secondary_degree",
    key_col: str = "key",
    last_chord_duration: float | None = None,
    drop_I_or_i: bool = True,
):
    """

    Args:
        last_chord_duration: The duration of the last chord. If "release" is in the
            DataFrame, this is ignored.

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,release,degree,key
    ... 0.0,0.5,I,C
    ... 0.5,1.0,V,C
    ... 1,2.0,V/V,C
    ... 2,3.0,I/V,C
    ... 3,4.0,V/V,C
    ... 4,5.0,I/V,C
    ... 5,6.0,V/V,C
    ... 6,7.0,I/V,C
    ... 7,8.0,V/V,C
    ... 8,9.0,V/vi,C
    ... 9,10.0,I/vi,C
    ... '''
    ...     )
    ... )
    >>> tonicization_census(df)
       chord_df_index  onset secondary_degree  n_chords  duration
    1               2    1.0                V         7       7.0
    2               9    8.0               vi         2       2.0
    >>> tonicization_census(df.drop(columns=["release"]), last_chord_duration=4.0)
       chord_df_index  onset secondary_degree  n_chords  duration
    1               2    1.0                V         7       7.0
    2               9    8.0               vi         2       5.0
    """
    if secondary_degree_col not in chord_df.columns:
        chord_df = split_degree_into_primary_and_secondary(
            chord_df,
            degree_col=degree_col,
            primary_degree_col=primary_degree_col,
            secondary_degree_col=secondary_degree_col,
        )
    tonicization_changes = (
        chord_df[secondary_degree_col] != chord_df[secondary_degree_col].shift(1)
    ) | (chord_df[key_col] != chord_df[key_col].shift(1))

    changes_df = pd.DataFrame(
        chord_df.loc[tonicization_changes, ["onset", secondary_degree_col]]
    )

    changes_df["n_chords"] = np.diff(
        np.concatenate([changes_df.index, [len(chord_df)]])
    )

    # We need to reset the index *after* computing the number of chords
    changes_df = changes_df.reset_index(names="chord_df_index")

    changes_df.loc[range(0, len(changes_df) - 1), "duration"] = np.diff(
        changes_df["onset"]
    )

    last_onset = chord_df.iloc[-1]["onset"]
    last_change_onset = changes_df.iloc[-1]["onset"]

    if "release" in chord_df.columns:
        last_release = chord_df.iloc[-1]["release"]
        last_duration = last_release - last_onset
    else:
        assert last_chord_duration is not None
        last_duration = last_chord_duration
    changes_df.loc[len(changes_df) - 1, "duration"] = (
        last_onset - last_change_onset + last_duration
    )
    if drop_I_or_i:
        changes_df = changes_df[~changes_df[secondary_degree_col].isin(["I", "i"])]

    return changes_df


def modulation_census(chord_df: pd.DataFrame, last_chord_duration: float | None = None):
    """

    Args:
        chord_df: A DataFrame with columns "onset", "degree", and "key" and possibly
            "release".
        last_chord_duration: The duration of the last chord. If "release" is in the
            DataFrame, this is ignored.

    >>> modulation_with_tonicization = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,release,degree,key
    ... 0.0,1.0,I,b
    ... 1.0,2.0,V/V,G
    ... 2.0,3.0,V,G
    ... 3.0,4.0,VI,b
    ... '''
    ...     )
    ... )
    >>> modulation_census(modulation_with_tonicization)
       chord_df_index  onset key  n_chords  duration
    0               0    0.0   b         1       1.0
    1               1    1.0   G         2       2.0
    2               3    3.0   b         1       1.0

    >>> modulation_census(
    ...     modulation_with_tonicization.drop(columns=["release"]),
    ...     last_chord_duration=4.0,
    ... )
       chord_df_index  onset key  n_chords  duration
    0               0    0.0   b         1       1.0
    1               1    1.0   G         2       2.0
    2               3    3.0   b         1       4.0
    """
    key_series = chord_df["key"].ffill()
    key_changes = key_series != key_series.shift(1)
    key_changes_df = pd.DataFrame(chord_df.loc[key_changes, ["onset", "key"]])
    key_changes_df["n_chords"] = np.diff(
        np.concatenate([key_changes_df.index, [len(chord_df)]])
    )
    key_changes_df = key_changes_df.reset_index(names="chord_df_index")
    key_changes_df.loc[range(0, len(key_changes_df) - 1), "duration"] = np.diff(
        key_changes_df["onset"]
    )

    last_onset = chord_df.iloc[-1]["onset"]
    last_change_onset = key_changes_df.iloc[-1]["onset"]
    if "release" in chord_df.columns:
        last_release = chord_df.iloc[-1]["release"]
        last_duration = last_release - last_onset
    else:
        assert last_chord_duration is not None
        last_duration = last_chord_duration
    key_changes_df.loc[len(key_changes_df) - 1, "duration"] = (
        last_onset - last_change_onset + last_duration
    )
    return key_changes_df
