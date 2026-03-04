import io  # noqa: F401
import warnings

import numpy as np
import pandas as pd

from music_df.chord_df import single_degree_to_split_degrees
from music_df.harmony.chords import (
    CacheDict,
    handle_nested_secondary_rns,
    rn_to_spelled_pitch,
    spelled_pitch_to_rn,
    tonicization_to_key,
)
from music_df.keys import get_key_sharps_interval


def assert_range_index(df: pd.DataFrame):
    try:
        assert isinstance(df.index, pd.RangeIndex)
    except AssertionError:
        assert (df.index == range(len(df))).all()
        return

    assert df.index.start == 0
    assert df.index.stop == len(df)
    assert df.index.step == 1


def _split_alteration(rn: str) -> tuple[str, str]:
    """Split 'bVII' -> ('b', 'VII'). No alteration -> ('_', 'VII')."""
    n = len(rn) - len(rn.lstrip("#b"))
    return (rn[:n] or "_", rn[n:])


def _warn_if_lowercase_degrees(df, cols=("primary_degree", "secondary_degree")):
    for col in cols:
        if col in df.columns and df[col].str.contains(r"[a-wy-z]", na=False).any():
            warnings.warn(
                f"Column '{col}' contains lowercase roman numerals. "
                "Use uppercase degrees with quality/secondary_mode columns instead.",
                FutureWarning,
                stacklevel=3,
            )


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
    ... primary_degree,secondary_degree,secondary_mode,quality,key
    ... VI,I,_,m,C
    ... V,VI,m,M,C
    ... VI,I,_,m,C
    ... II,VI,m,m,C
    ... V,VI,m,M,C
    ... VI,I,_,m,C
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df, quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0              I               VI              m       m   C
    1              V               VI              m       M   C
    2              I               VI              m       m   C
    3             II               VI              m       m   C
    4              V               VI              m       M   C
    5              I               VI              m       m   C

    Across key-change
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,secondary_mode,quality,key
    ... VI,I,_,m,C
    ... V,VI,m,M,G
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df, quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0             VI                I              _       m   C
    1              V               VI              m       M   G
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,secondary_mode,quality,key
    ... IV,II,m,M,Ab
    ... II,I,_,m,c
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df, quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0             IV               II              m       M  Ab
    1             II                I              _       m   c

    Mode-aware matching: quality "m" matches secondary_mode "m",
    quality "M" matches secondary_mode "M"
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,secondary_mode,quality,key
    ... VI,I,_,m,C
    ... V,VI,m,M,C
    ... VI,I,_,m,C
    ... IV,I,_,M,C
    ... V,VI,M,M,C
    ... VI,I,_,M,C
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df, quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0              I               VI              m       m   C
    1              V               VI              m       M   C
    2              I               VI              m       m   C
    3             IV                I              _       M   C
    4              V               VI              M       M   C
    5              I               VI              M       M   C

    Mode mismatch prevents expansion: quality "m" != secondary_mode "M"
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,secondary_mode,quality,key
    ... VI,I,_,m,C
    ... V,VI,M,M,C
    ... VI,I,_,m,C
    ... II,VI,M,m,C
    ... V,VI,M,M,C
    ... VI,I,_,M,C
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df, quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0             VI                I              _       m   C
    1              V               VI              M       M   C
    2             VI                I              _       m   C
    3             II               VI              M       m   C
    4              V               VI              M       M   C
    5              I               VI              M       M   C

    Dominant chord behavior
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,secondary_mode,quality,key
    ... V,V,M,M,Ab
    ... V,I,_,M,Ab
    ... V,V,M,M,Ab
    ... '''
    ...     )
    ... )

    When V is preceded and followed by tonicization, it is expanded
    >>> expand_tonicizations(df.copy(), quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0              V                V              M       M  Ab
    1              I                V              M       M  Ab
    2              V                V              M       M  Ab

    When V is only preceded or only followed by tonicization, it is not expanded
    >>> expand_tonicizations(df.iloc[1:].copy(), quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    1              V                I              _       M  Ab
    2              V                V              M       M  Ab
    >>> expand_tonicizations(df.iloc[:-1].copy(), quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0              V                V              M       M  Ab
    1              V                I              _       M  Ab

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,secondary_degree,secondary_mode,quality,key
    ... V,V,M,M,Ab
    ... V,I,_,Mm7,Ab
    ... V,V,M,M,Ab
    ... V,I,_,aug6,Ab
    ... V,V,M,M,Ab
    ... V,I,_,o,Ab
    ... V,V,M,M,Ab
    ... V,I,_,+,Ab
    ... V,V,M,M,Ab
    ... '''
    ...     )
    ... )
    >>> expand_tonicizations(df, quality_col="quality")
      primary_degree secondary_degree secondary_mode quality key
    0              V                V              M       M  Ab
    1              V                I              _     Mm7  Ab
    2              V                V              M       M  Ab
    3              V                I              _    aug6  Ab
    4              V                V              M       M  Ab
    5              V                I              _       o  Ab
    6              V                V              M       M  Ab
    7              V                I              _       +  Ab
    8              V                V              M       M  Ab
    """

    _warn_if_lowercase_degrees(df)

    _ALT_SENTINELS = {"_", "-", ""}

    def _alts_match(shift_n):
        if (
            "primary_alteration" not in df.columns
            or "secondary_alteration" not in df.columns
        ):
            return True
        return df["primary_alteration"].isin(_ALT_SENTINELS) & df[
            "secondary_alteration"
        ].shift(shift_n).isin(_ALT_SENTINELS) | (
            df["primary_alteration"] == df["secondary_alteration"].shift(shift_n)
        )

    def _modes_match(shift_n):
        if quality_col is None or "secondary_mode" not in df.columns:
            return True
        quality_mode = df[quality_col].map({"M": "M", "m": "m"})
        return quality_mode == df["secondary_mode"].shift(shift_n)

    non_dominant_mask = (
        (df["primary_degree"] != "V")
        & (df["secondary_degree"] == "I")
        & (
            (
                (df["primary_degree"] == df["secondary_degree"].shift(-1))
                & (df["key"] == df["key"].shift(-1))
                & _alts_match(-1)
                & _modes_match(-1)
            )
            | (
                (df["primary_degree"] == df["secondary_degree"].shift(1))
                & (df["key"] == df["key"].shift(1))
                & _alts_match(1)
                & _modes_match(1)
            )
        )
    )

    dominant_mask = (
        (df["primary_degree"] == "V")
        & (df["secondary_degree"] == "I")
        & (
            (df["primary_degree"] == df["secondary_degree"].shift(-1))
            & (df["key"] == df["key"].shift(-1))
            & _alts_match(-1)
            & _modes_match(-1)
            & (df["primary_degree"] == df["secondary_degree"].shift(1))
            & (df["key"] == df["key"].shift(1))
            & _alts_match(1)
            & _modes_match(1)
        )
    )

    if quality_col is not None:
        non_dominant_mask &= ~df[quality_col].str.contains("Mm7|aug6|o|\\+")
        dominant_mask &= ~df[quality_col].str.contains("Mm7|aug6|o|\\+")

    def _apply_mask(mask):
        df.loc[mask, "secondary_degree"] = df.loc[mask, "primary_degree"]

        if "secondary_mode" in df.columns:
            if quality_col is not None:
                quality_mode = df.loc[mask, quality_col].map({"M": "M", "m": "m"})
                df.loc[mask, "secondary_mode"] = quality_mode
            else:
                upper_idx = df.loc[mask].index[
                    df.loc[mask, "primary_degree"].str[-1].str.isupper()
                ]
                lower_idx = df.loc[mask].index[
                    df.loc[mask, "primary_degree"].str[-1].str.islower()
                ]
                df.loc[upper_idx, "secondary_mode"] = "M"
                df.loc[lower_idx, "secondary_mode"] = "m"

        if "primary_alteration" in df.columns and "secondary_alteration" in df.columns:
            df.loc[mask, "secondary_alteration"] = df.loc[mask, "primary_alteration"]
            df.loc[mask, "primary_alteration"] = "_"

        df.loc[mask, "primary_degree"] = "I"

    _apply_mask(non_dominant_mask)
    _apply_mask(dominant_mask)

    return df



def _reconstruct_degree_column(chord_df: pd.DataFrame) -> pd.Series:
    _ALT_SENTINELS = {"_", "-"}

    def _alt_col(col_name):
        if col_name in chord_df.columns:
            return chord_df[col_name].replace(list(_ALT_SENTINELS), "")
        return ""

    primary = _alt_col("primary_alteration") + chord_df["primary_degree"]

    secondary_with_slash = (
        "/" + _alt_col("secondary_alteration") + chord_df["secondary_degree"]
    )
    if "secondary_mode" in chord_df.columns:
        mode_suffix = chord_df["secondary_mode"].fillna("_").replace("_", "")
        secondary_with_slash = secondary_with_slash + mode_suffix

    # Build a mask of rows where the secondary part is redundant ("/I" in the
    # current key's mode) and should be stripped.
    is_I = chord_df["secondary_degree"] == "I"
    if "secondary_alteration" in chord_df.columns:
        is_I = is_I & chord_df["secondary_alteration"].isin(_ALT_SENTINELS)

    if "key" in chord_df.columns and "secondary_mode" in chord_df.columns:
        key_mode = chord_df["key"].str[0].map(
            lambda c: "M" if c.isupper() else "m"
        )
        mode_val = chord_df["secondary_mode"].fillna("_")
        strip_mask = is_I & ((mode_val == "_") | (mode_val == key_mode))
    elif "key" in chord_df.columns:
        strip_mask = is_I
    else:
        strip_mask = is_I

    secondary_with_slash = secondary_with_slash.where(~strip_mask, "")
    return primary + secondary_with_slash


def _harmony_group_ids(chord_df: pd.DataFrame) -> pd.Series:
    """Assign integer group IDs to consecutive runs of identical harmonies.

    Used for de-repeating (label-based comparison before expand_tonicizations).
    Compares all harmony-relevant columns, intentionally excluding ``inversion``
    so that different inversions of the same chord belong to the same group.
    """
    cols = ["primary_degree", "secondary_degree", "key"]
    optional = ["primary_alteration", "secondary_alteration", "secondary_mode", "quality"]
    cols += [c for c in optional if c in chord_df.columns]

    same_as_prev = pd.Series(True, index=chord_df.index)
    for col in cols:
        same_as_prev &= chord_df[col] == chord_df[col].shift(1)
    first_of_group = ~same_as_prev
    # shift(1) produces NA for the first row; with PyArrow-backed dtypes,
    # ~NA stays NA rather than becoming True. The first row is always a group start.
    first_of_group.iloc[0] = True
    return first_of_group.astype(int).cumsum() - 1


def _count_group_ids(chord_df: pd.DataFrame) -> pd.Series:
    """Assign group IDs using pitch-class set subsumption when ``chord_pcs``
    is available, falling back to :func:`_harmony_group_ids` otherwise.

    Two adjacent chords in the same key are considered the same harmony if
    either chord's PC set is a subset of the other's (e.g., M {0,4,7} ⊂ Mm7
    {0,4,7,a}). When merged, the **superset** is propagated so that later
    comparisons use the richer chord.
    """
    if "chord_pcs" not in chord_df.columns:
        return _harmony_group_ids(chord_df)

    group_ids = []
    group_id = 0
    propagated_pcs: frozenset[str] = frozenset()
    prev_key = None

    for i, row in enumerate(chord_df.itertuples()):
        current_pcs = frozenset(row.chord_pcs)
        current_key = row.key

        if i == 0:
            group_ids.append(group_id)
            propagated_pcs = current_pcs
            prev_key = current_key
            continue

        if current_key != prev_key:
            group_id += 1
            propagated_pcs = current_pcs
        elif current_pcs <= propagated_pcs or propagated_pcs <= current_pcs:
            # Subsumption in either direction; propagate the superset
            propagated_pcs = propagated_pcs | current_pcs
        else:
            group_id += 1
            propagated_pcs = current_pcs

        prev_key = current_key
        group_ids.append(group_id)

    result = pd.Series(group_ids, index=chord_df.index)
    assert result.dtype == int
    assert result.iloc[0] == 0
    return result


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

    Note that repeated harmonies (e.g., "V" followed by "V" with a different
    inversion) are handled by de-repeating before expanding tonicizations.

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
        max_tonicization_num_chords: The maximum number of distinct chords in a
            tonicization. At least one of max_tonicization_duration or
            max_tonicization_num_chords must be provided. Different inversions of
            the same chord count as one; consecutive identical harmonies are
            de-repeated before counting.
        min_removal_num_chords: If provided, a tonicization must have at least this
            number of distinct chords to be removed. Counting follows the same
            inversion-aware rules as max_tonicization_num_chords. This argument
            must be <= max_tonicization_num_chords if both are provided.
        tonicization_cache: A cache for tonicizations to save compute if running
            this function many times.
        case_matters: Whether to consider case when determining the key of the
            tonicization. Ignored if tonicization_cache is provided.
        simplify_enharmonics: Whether to simplify enharmonic spellings of the key of
            the tonicization. Ignored if tonicization_cache is provided.

    >>> test_case = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... chord_pcs,onset,release,inversion,key,quality,primary_degree,primary_alteration,secondary_degree,secondary_alteration,secondary_mode
    ... 590,374.0,375.0,0.0,F,M,I,_,I,_,
    ... 038,393.0,394.0,1.0,F,M,I,_,III,b,M
    ... 37a,394.0,397.0,0.0,F,M,V,_,III,b,M
    ... 36b,405.0,406.0,1.0,F,M,I,_,IV,#,M
    ... 6a1,406.0,409.0,0.0,F,M,V,_,IV,#,M
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(test_case, max_tonicization_num_chords=1)
      chord_pcs  onset  release  inversion key quality primary_degree primary_alteration secondary_degree secondary_alteration secondary_mode degree
    0       590  374.0    375.0        0.0   F       M              I                  _                I                    _              _      I
    1       038  393.0    394.0        1.0  Ab       M              I                  _                I                    -              _      I
    2       37a  394.0    397.0        0.0  Ab       M              V                  _                I                    -              _      V
    3       36b  405.0    406.0        1.0   B       M              I                  _                I                    -              _      I
    4       6a1  406.0    409.0        0.0   B       M              V                  _                I                    -              _      V

    >>> no_repeat = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,V/V,C
    ... 4.0,V,C
    ... 8.0,V/V,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(no_repeat, max_tonicization_duration=4)
       onset degree key
    0    0.0      V   G
    1    4.0      I   G
    2    8.0      V   G

    >>> yes_repeat = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,inversion,key
    ... 0.0,V/V,0,C
    ... 4.0,V,0,C
    ... 6.0,V,1,C
    ... 8.0,V/V,0,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(yes_repeat, max_tonicization_duration=4)
       onset degree  inversion key
    0    0.0      V          0   G
    1    4.0      I          0   G
    2    6.0      I          1   G
    3    8.0      V          0   G

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
    ... 1.0,II/IV,bb
    ... 2.0,V/IV,bb
    ... 3.0,II/III,eb
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
    1    1.0     II  eb
    2    2.0      V  eb
    3    3.0     II  Gb
    4    4.0      V  Gb
    5    5.0      I  eb

    >>> tonicization_at_beginning_and_end = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,V/V,C
    ... 1.0,VIIo/V,C
    ... 2.0,I,C
    ... 3.0,IV/II,C
    ... 4.0,V/II,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(
    ...     tonicization_at_beginning_and_end, max_tonicization_num_chords=1
    ... )
       onset degree key
    0    0.0      V   G
    1    1.0   VIIo   G
    2    2.0      I   C
    3    3.0     IV   d
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
    ... 1.0,II/V,C
    ... 1.125,V/V,C
    ... 1.25,II/V,C
    ... 1.375,V/V,C
    ... 1.5,V,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(multiple_brief_chords, max_tonicization_num_chords=1)
       onset degree key
    0  0.000      I   C
    1  1.000     II   G
    2  1.125      V   G
    3  1.250     II   G
    4  1.375      V   G
    5  1.500      V   C
    >>> remove_long_tonicizations(
    ...     multiple_brief_chords,
    ...     max_tonicization_num_chords=1,
    ...     min_removal_duration=1,
    ... )
       onset degree key
    0  0.000      I   C
    1  1.000   II/V   C
    2  1.125    V/V   C
    3  1.250   II/V   C
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
    0    0.0    V/V   C
    1    1.0    V/V   C
    2    2.0    V/V   C
    3    3.0    V/V   C
    >>> remove_long_tonicizations(consecutive_chords, max_tonicization_num_chords=0)
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

    Inversions within a tonicization count as one chord (1 distinct chord
    <= 2, so the tonicization is kept):
    >>> inversions = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,inversion,key
    ... 0.0,I,0,C
    ... 1.0,V/V,0,C
    ... 2.0,V/V,1,C
    ... 3.0,V/V,2,C
    ... 4.0,I,0,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(inversions, max_tonicization_num_chords=2)
       onset degree  inversion key
    0    0.0      I          0   C
    1    1.0    V/V          0   C
    2    2.0    V/V          1   C
    3    3.0    V/V          2   C
    4    4.0      I          0   C

    Different qualities are NOT collapsed during de-repeat, so V(M) and
    V(Mm7) count as 2 distinct chords (> 1 → removed):
    >>> qualities = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,primary_degree,secondary_degree,quality,key
    ... 0.0,I,I,M,C
    ... 1.0,V,V,M,C
    ... 2.0,V,V,Mm7,C
    ... 3.0,I,I,M,C
    ... '''
    ...     )
    ... )
    >>> remove_long_tonicizations(qualities, max_tonicization_num_chords=1)
       onset primary_degree secondary_degree quality key degree
    0    0.0              I                I       M   C      I
    1    1.0              V                I       M   G      V
    2    2.0              V                I     Mm7   G      V
    3    3.0              I                I       M   C      I

    """
    _warn_if_lowercase_degrees(chord_df)

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
        chord_df = single_degree_to_split_degrees(chord_df)
    else:
        raise ValueError(
            f"chord_df must have columns {split_columns} or {joined_columns}"
        )

    # We begin by filling the key column so that if we change the key during a
    # tonicization, it will revert to the original key after the tonicized passage.

    chord_df["key"] = chord_df["key"].ffill()

    # We fill the secondary_degree column with "I"
    chord_df["secondary_degree"] = chord_df["secondary_degree"].fillna("I")
    if "secondary_mode" in chord_df.columns:
        chord_df["secondary_mode"] = chord_df["secondary_mode"].fillna("_")

    # De-repeat for expand_tonicizations: consecutive rows with the same
    # harmony (including quality) confuse the shift-based neighbor checks.
    # We collapse them, expand, then map results back.
    group_ids = _harmony_group_ids(chord_df)
    first_of_group = group_ids != group_ids.shift(1)
    first_of_group.iloc[0] = True

    derepeated = chord_df.loc[first_of_group].copy().reset_index(drop=True)
    derepeated = expand_tonicizations(
        derepeated, quality_col="quality" if "quality" in derepeated.columns else None
    )

    cols_to_map = ["primary_degree", "secondary_degree"]
    if "secondary_mode" in chord_df.columns:
        cols_to_map.append("secondary_mode")
    if "secondary_alteration" in chord_df.columns:
        cols_to_map.append("secondary_alteration")
    for col in cols_to_map:
        chord_df[col] = group_ids.map(derepeated[col]).values

    # Post-expansion group IDs for counting: uses PC-set subsumption when
    # chord_pcs is available, otherwise label-based comparison.
    count_group_ids = _count_group_ids(chord_df)

    tonicization_changes = (
        chord_df["secondary_degree"] != chord_df["secondary_degree"].shift(1)
    ) | (chord_df["key"] != chord_df["key"].shift(1))
    if "secondary_alteration" in chord_df.columns:
        tonicization_changes |= (
            chord_df["secondary_alteration"]
            != chord_df["secondary_alteration"].shift(1)
        )

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
        full_tonicization = tonicization
        if "secondary_alteration" in chord_df.columns:
            alt = chord_df.iloc[start_i]["secondary_alteration"]
            if pd.notna(alt) and alt not in ("_", "-", ""):
                full_tonicization = alt + tonicization
        new_key = get_tonicized_key(
            full_tonicization, chord_df.iloc[start_i]["key"], secondary_mode
        )
        chord_df.loc[start_i : end_i - 1, "key"] = new_key
        chord_df.loc[start_i : end_i - 1, "secondary_degree"] = "I"
        if "secondary_alteration" in chord_df.columns:
            chord_df.loc[start_i : end_i - 1, "secondary_alteration"] = "-"
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

        n_distinct = count_group_ids.iloc[start_i:end_i].nunique()

        if (
            max_tonicization_num_chords is not None
            and n_distinct > max_tonicization_num_chords
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
                or n_distinct >= min_removal_num_chords
            ):
                remove_tonicization(tonicization, start_i, end_i)

    chord_df = replace_spurious_tonics(chord_df)

    # Drop secondary_mode before reconstruction if it wasn't in the original input,
    # so mode suffixes don't leak into the degree column.
    if not had_secondary_mode and "secondary_mode" in chord_df.columns:
        chord_df = chord_df.drop(columns=["secondary_mode"])

    chord_df["degree"] = _reconstruct_degree_column(chord_df)

    if not had_split_columns:
        chord_df = chord_df.drop(
            columns=[
                "primary_degree",
                "secondary_degree",
                "primary_alteration",
                "secondary_alteration",
            ],
            errors="ignore",
        )

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
    ].str.upper()
    chord_df.loc[spurious_tonic_mask, "secondary_degree"] = "I"
    if (
        "secondary_alteration" in chord_df.columns
        and "primary_alteration" in chord_df.columns
    ):
        chord_df.loc[spurious_tonic_mask, "primary_alteration"] = chord_df.loc[
            spurious_tonic_mask, "secondary_alteration"
        ]
        chord_df.loc[spurious_tonic_mask, "secondary_alteration"] = "_"
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

    ``min_modulation_num_chords`` and ``max_removal_num_chords`` count distinct
    harmonies: different inversions of the same chord count as one.


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
    ... 1.25,VIIo6,C
    ... 1.5,I,C
    ... 2.0,I,Ab
    ... '''
    ...     )
    ... )
    >>> remove_short_modulations(brief_modulation, min_modulation_duration=2.0)
       onset     degree key
    0   0.00          I  Ab
    1   1.00        III  Ab
    2   1.25  VIIo6/III  Ab
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
    _warn_if_lowercase_degrees(chord_df)

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
        chord_df = single_degree_to_split_degrees(chord_df)
    else:
        raise ValueError(
            f"chord_df must have columns {split_columns} or {joined_columns}"
        )

    chord_df["key"] = chord_df["key"].ffill()

    # Counting: uses PC-set subsumption when chord_pcs is available,
    # otherwise label-based comparison.
    count_group_ids = _count_group_ids(chord_df)

    key_changes = chord_df["key"] != chord_df["key"].shift(1)
    indices = key_changes.index[key_changes].tolist()
    indices.append(len(chord_df))

    def get_secondary_rn(inner_key: str, outer_key: str):
        inner_key_mode = "M" if inner_key[0].isupper() else "m"

        if spelled_pitch_to_rn_cache is None:
            return spelled_pitch_to_rn(inner_key, outer_key, inner_key_mode)
        else:
            return spelled_pitch_to_rn_cache[(inner_key, outer_key, inner_key_mode)]

    def nested_handler(row: pd.Series, outer_key: str):
        # Resolve the inner secondary degree to a pitch, then express
        # relative to the outer key. We can't just concatenate
        # row["secondary_degree"] + "/" + secondary_degree and use
        # handle_nested_secondary_rns because the two parts live in
        # different reference frames (inner key vs outer key).
        sec_deg = row["secondary_degree"]
        if "secondary_alteration" in row.index:
            alt = row["secondary_alteration"]
            if alt not in ("_", "-", ""):
                sec_deg = alt + sec_deg
        inner_pitch = rn_to_spelled_pitch(sec_deg, row["key"])
        if spelled_pitch_to_rn_cache is not None:
            return spelled_pitch_to_rn_cache[(inner_pitch, outer_key, "M")]
        return spelled_pitch_to_rn(inner_pitch, outer_key, "M")

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
        if len(nested_secondary_index) > 0:
            nested_results = chord_df.loc[nested_secondary_index].apply(
                lambda row: nested_handler(row, outer_key), axis=1
            )
            for idx, rn in nested_results.items():
                alt, bare = _split_alteration(rn)
                if "secondary_mode" in chord_df.columns:
                    collapsed_mode = "M" if bare[-1].isupper() else "m"
                    chord_df.loc[idx, "secondary_mode"] = collapsed_mode
                bare = bare.upper()
                chord_df.loc[idx, "secondary_degree"] = bare
                if "secondary_alteration" in chord_df.columns:
                    chord_df.loc[idx, "secondary_alteration"] = alt

        # Handle simple secondary RNs
        simple_secondary_index = nested_secondary_mask.index[~nested_secondary_mask]
        sec_alt, sec_bare = _split_alteration(secondary_degree)
        sec_bare = sec_bare.upper()
        chord_df.loc[simple_secondary_index, "secondary_degree"] = sec_bare
        if "secondary_alteration" in chord_df.columns:
            chord_df.loc[simple_secondary_index, "secondary_alteration"] = sec_alt

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

        n_distinct = count_group_ids.iloc[start_i:end_i].nunique()

        if (
            min_modulation_num_chords is not None
            and n_distinct < min_modulation_num_chords
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
                or n_distinct <= max_removal_num_chords
            ):
                remove_modulation_if_possible(start_i, end_i)

    replace_spurious_tonics(chord_df, inplace=True)

    if not had_secondary_mode and "secondary_mode" in chord_df.columns:
        chord_df = chord_df.drop(columns=["secondary_mode"])

    chord_df["degree"] = _reconstruct_degree_column(chord_df)

    if not had_split_columns:
        chord_df = chord_df.drop(
            columns=[
                "primary_degree",
                "secondary_degree",
                "primary_alteration",
                "secondary_alteration",
            ],
            errors="ignore",
        )

    return chord_df


def remove_phantom_keys(
    chord_df: pd.DataFrame,
    inplace: bool = False,
    tonicization_cache: CacheDict[tuple[str, str, str | None], str] | None = None,
    spelled_pitch_to_rn_cache: CacheDict[tuple[str, str, str], str] | None = None,
    case_matters: bool = False,
    simplify_enharmonics: bool = True,
) -> pd.DataFrame:
    """Remove phantom keys by absorbing them into the nearest neighbor.

    After removing tonicizations or short modulations, some key segments may
    have every chord tonicized (secondary_degree != "I"/"i"). These "phantom"
    keys are absorbed into the nearest neighboring key on the circle of fifths.
    When a phantom region contains multiple distinct tonicized keys, the function
    finds an optimal contiguous split point assigning blocks to the preceding
    or following neighbor.

    >>> absorbed_into_following = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... 1.0,V/IV,D
    ... 2.0,I,G
    ... '''
    ...     )
    ... )
    >>> remove_phantom_keys(absorbed_into_following)
       onset degree key
    0    0.0      I   C
    1    1.0      V   G
    2    2.0      I   G

    >>> absorbed_into_preceding = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... 1.0,V/IV,G
    ... 2.0,II/IV,G
    ... 3.0,I,A
    ... '''
    ...     )
    ... )
    >>> remove_phantom_keys(absorbed_into_preceding)
       onset degree key
    0    0.0      I   C
    1    1.0      V   C
    2    2.0     II   C
    3    3.0      I   A

    >>> not_phantom = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,degree,key
    ... 0.0,I,C
    ... 1.0,V/V,G
    ... 2.0,IV,G
    ... 3.0,I,D
    ... '''
    ...     )
    ... )
    >>> remove_phantom_keys(not_phantom)
       onset degree key
    0    0.0      I   C
    1    1.0    V/V   G
    2    2.0     IV   G
    3    3.0      I   D
    """
    _warn_if_lowercase_degrees(chord_df)

    if not inplace:
        chord_df = chord_df.copy()

    assert_range_index(chord_df)

    split_columns = ["primary_degree", "secondary_degree", "key", "onset"]
    joined_columns = ["onset", "degree", "key"]
    had_secondary_mode = "secondary_mode" in chord_df.columns
    if all(k in chord_df.columns for k in split_columns):
        had_split_columns = True
    elif all(k in chord_df.columns for k in joined_columns):
        had_split_columns = False
        chord_df = single_degree_to_split_degrees(chord_df)
    else:
        raise ValueError(
            f"chord_df must have columns {split_columns} or {joined_columns}"
        )

    chord_df["key"] = chord_df["key"].ffill()
    chord_df["secondary_degree"] = chord_df["secondary_degree"].fillna("I")
    if "secondary_mode" in chord_df.columns:
        chord_df["secondary_mode"] = chord_df["secondary_mode"].fillna("_")

    # Identify key segments
    key_changes = chord_df["key"] != chord_df["key"].shift(1)
    change_indices = key_changes.index[key_changes].tolist()
    change_indices.append(len(chord_df))

    segments = []
    for start_i, end_i in zip(change_indices[:-1], change_indices[1:]):
        key = chord_df.loc[start_i, "key"]
        segments.append((start_i, end_i, key))

    # Classify: phantom if no chord has secondary_degree "I"
    is_phantom = []
    for start_i, end_i, _key in segments:
        sec_deg = chord_df.loc[start_i : end_i - 1, "secondary_degree"]
        is_phantom.append(not (sec_deg == "I").any())

    def _get_tonicized_key(secondary_degree, key, secondary_mode=None):
        if tonicization_cache is None:
            return tonicization_to_key(
                secondary_degree,
                key,
                case_matters,
                simplify_enharmonics,
                secondary_mode,
            )
        return tonicization_cache[(secondary_degree, key, secondary_mode)]

    def _get_rn(tk_pitch, neighbor_key, tk_mode):
        if spelled_pitch_to_rn_cache is None:
            return spelled_pitch_to_rn(tk_pitch, neighbor_key, tk_mode)
        return spelled_pitch_to_rn_cache[(tk_pitch, neighbor_key, tk_mode)]

    # Process consecutive phantom regions
    i = 0
    while i < len(segments):
        if not is_phantom[i]:
            i += 1
            continue

        # Find extent of consecutive phantom segments
        j = i
        while j < len(segments) and is_phantom[j]:
            j += 1

        prev_key = segments[i - 1][2] if i > 0 else None
        next_key = segments[j][2] if j < len(segments) else None

        if prev_key is None and next_key is None:
            i = j
            continue

        region_start = segments[i][0]
        region_end = segments[j - 1][1]

        # Compute tonicized key for each chord
        tonicized_keys = []
        for idx in range(region_start, region_end):
            sec_deg = chord_df.loc[idx, "secondary_degree"]
            chord_key = chord_df.loc[idx, "key"]
            sec_mode = None
            if "secondary_mode" in chord_df.columns:
                sm = chord_df.loc[idx, "secondary_mode"]
                if sm in ("M", "m"):
                    sec_mode = sm
            tonicized_keys.append(_get_tonicized_key(sec_deg, chord_key, sec_mode))

        # Group into blocks of consecutive same tonicized key
        blocks = []  # (block_start_idx, block_end_idx, tonicized_key)
        block_start = 0
        for k in range(1, len(tonicized_keys)):
            if tonicized_keys[k] != tonicized_keys[k - 1]:
                blocks.append(
                    (
                        region_start + block_start,
                        region_start + k,
                        tonicized_keys[block_start],
                    )
                )
                block_start = k
        blocks.append(
            (
                region_start + block_start,
                region_end,
                tonicized_keys[block_start],
            )
        )

        # Find optimal split: k blocks to preceding, rest to following.
        # Minimize total cost = sum(n_chords * circle-of-fifths distance).
        # Ties prefer preceding (larger k) via <=.
        if prev_key is None:
            split_k = 0
        elif next_key is None:
            split_k = len(blocks)
        else:
            best_cost = float("inf")
            best_k = 0
            for k in range(len(blocks) + 1):
                cost = 0
                for bi in range(k):
                    n = blocks[bi][1] - blocks[bi][0]
                    cost += n * abs(get_key_sharps_interval(prev_key, blocks[bi][2]))
                for bi in range(k, len(blocks)):
                    n = blocks[bi][1] - blocks[bi][0]
                    cost += n * abs(get_key_sharps_interval(next_key, blocks[bi][2]))
                if cost <= best_cost:
                    best_cost = cost
                    best_k = k
            split_k = best_k

        # Re-express secondary degrees relative to the assigned neighbor
        for bi, (blk_start, blk_end, tk) in enumerate(blocks):
            neighbor_key = prev_key if bi < split_k else next_key
            tk_pitch = tk[0].upper() + tk[1:]
            tk_mode = "M" if tk[0].isupper() else "m"
            new_sec = _get_rn(tk_pitch, neighbor_key, tk_mode)
            new_sec_alt, new_sec_bare = _split_alteration(new_sec)
            new_sec_bare = new_sec_bare.upper()
            mode = tk_mode
            if new_sec_bare == "I":
                mode = "_"
                new_sec_alt = "_"

            chord_df.loc[blk_start : blk_end - 1, "secondary_degree"] = new_sec_bare
            if "secondary_alteration" in chord_df.columns:
                chord_df.loc[blk_start : blk_end - 1, "secondary_alteration"] = (
                    new_sec_alt
                )
            if "secondary_mode" in chord_df.columns:
                chord_df.loc[blk_start : blk_end - 1, "secondary_mode"] = mode
            chord_df.loc[blk_start : blk_end - 1, "key"] = neighbor_key

        i = j

    replace_spurious_tonics(chord_df, inplace=True)

    if not had_secondary_mode and "secondary_mode" in chord_df.columns:
        chord_df = chord_df.drop(columns=["secondary_mode"])

    chord_df["degree"] = _reconstruct_degree_column(chord_df)

    if not had_split_columns:
        chord_df = chord_df.drop(
            columns=[
                "primary_degree",
                "secondary_degree",
                "primary_alteration",
                "secondary_alteration",
            ],
            errors="ignore",
        )

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
    ... 8,9.0,V/VI,C
    ... 9,10.0,I/VI,C
    ... '''
    ...     )
    ... )
    >>> tonicization_census(df)
       chord_df_index  onset secondary_degree  n_chords  duration
    1               2    1.0                V         7       7.0
    2               9    8.0               VI         2       2.0
    >>> tonicization_census(df.drop(columns=["release"]), last_chord_duration=4.0)
       chord_df_index  onset secondary_degree  n_chords  duration
    1               2    1.0                V         7       7.0
    2               9    8.0               VI         2       5.0
    """
    if secondary_degree_col not in chord_df.columns:
        chord_df = single_degree_to_split_degrees(
            chord_df,
            degree_col=degree_col,
            primary_degree_col=primary_degree_col,
            secondary_degree_col=secondary_degree_col,
        )
    tonicization_changes = (
        chord_df[secondary_degree_col] != chord_df[secondary_degree_col].shift(1)
    ) | (chord_df[key_col] != chord_df[key_col].shift(1))
    if "secondary_alteration" in chord_df.columns:
        tonicization_changes |= (
            chord_df["secondary_alteration"]
            != chord_df["secondary_alteration"].shift(1)
        )

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
