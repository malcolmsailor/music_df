"""
Provides a function to "dedouble" a music_df.

Dedoubling is removing rows that have the same type, onset, release, and pitch.
"""

from typing import Iterable, Sequence

import numpy as np  # Used by doctests
import pandas as pd

from music_df.dedouble_instruments import CANDIDATE_INSTRUMENT_COLUMNS
from music_df.transforms import transform


@transform
def dedouble(
    df: pd.DataFrame,
    quantize: bool = True,
    ticks_per_quarter: int = 16,
    release_ticks_per_quarter: int | None = None,
    match_releases: bool = True,
    # columns are always deduplicated on onset and release
    columns: Iterable[str] = ("type", "pitch"),
    instrument_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Dedouble a music_df.

    Args:
        df: The dataframe to dedouble.
        quantize: Whether to quantize the onset and release times before dedoubling.
            Default is True.
        ticks_per_quarter: The number of ticks per quarter note when quantizing.
            Default is 16.
        release_ticks_per_quarter: Grid resolution for releases when quantizing.
            If None, uses ticks_per_quarter.
        match_releases: Whether to match releases as well as onsets when dedoubling.
            Default is True.
        columns: The columns, other than onset and release, to deduplicate on.
            Default is ("type", "pitch").
        instrument_columns: Columns that identify an instrument. If None,
            auto-detected from CANDIDATE_INSTRUMENT_COLUMNS. Any found columns
            are added to the dedup grouping so notes are only deduplicated
            within the same instrument.

    >>> df = pd.DataFrame(
    ...     {
    ...         "type": ["time_signature", "bar", "note", "note", "note", "bar"],
    ...         "track": [0.0, np.nan, 1.0, 1.0, 2.0, np.nan],
    ...         "channel": [np.nan, np.nan, 0.0, 0.0, 0.0, np.nan],
    ...         "pitch": [np.nan, np.nan, 50.0, 50.0, 50.0, np.nan],
    ...         "onset": [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    ...         "release": [np.nan, np.nan, 1.0, 0.99, 1.0, np.nan],
    ...         "velocity": [np.nan, np.nan, 100.0, 100.0, 100.0, np.nan],
    ...         "other": [
    ...             {"numerator": 3, "denominator": 4, "clocks_per_click": None},
    ...             np.nan,
    ...             np.nan,
    ...             np.nan,
    ...             np.nan,
    ...             np.nan,
    ...         ],
    ...     }
    ... )
    >>> pd.set_option(
    ...     "display.width", 200
    ... )  # To avoid issues when the terminal is a different size
    >>> pd.set_option("display.max_columns", None)

    >>> df  # doctest: +NORMALIZE_WHITESPACE
                 type  track  channel  pitch  onset  release  velocity                                              other
    0  time_signature    0.0      NaN    NaN    0.0      NaN       NaN  {'numerator': 3, 'denominator': 4, 'clocks_per...
    1             bar    NaN      NaN    NaN    0.0      NaN       NaN                                                NaN
    2            note    1.0      0.0   50.0    0.0     1.00     100.0                                                NaN
    3            note    1.0      0.0   50.0    0.0     0.99     100.0                                                NaN
    4            note    2.0      0.0   50.0    0.0     1.00     100.0                                                NaN
    5             bar    NaN      NaN    NaN    3.0      NaN       NaN                                                NaN

    Auto-detects "track" and "channel" columns, so notes on different
    tracks are preserved (rows 2 and 4 have different tracks):
    >>> dedouble(df)[
    ...     ["original_index", "duplicated_indices"]
    ... ]  # doctest: +NORMALIZE_WHITESPACE
       original_index duplicated_indices
    0               0
    1               1
    2               2                2,3
    3               4
    4               5

    We store the following attributes in .attrs:
    - dedoubled (bool)
    - n_dedoubled_notes (int)
    - n_undedoubled_notes (int)
    >>> dedouble(df).attrs
    {'n_undedoubled_notes': 3, 'n_dedoubled_notes': 2, 'dedoubled': True}

    Without quantization, the release mismatch between rows 2 and 3
    prevents them from being grouped:
    >>> dedouble(df, quantize=False)[
    ...     ["original_index", "duplicated_indices"]
    ... ]  # doctest: +NORMALIZE_WHITESPACE
       original_index duplicated_indices
    0               0
    1               1
    2               2
    3               3
    4               4
    5               5

    Passing instrument_columns=() disables auto-detection, so all notes
    with the same pitch/onset/release are grouped regardless of track:
    >>> dedouble(df, instrument_columns=())[
    ...     ["original_index", "duplicated_indices"]
    ... ]  # doctest: +NORMALIZE_WHITESPACE
       original_index duplicated_indices
    0               0
    1               1
    2               2              2,3,4
    3               5

    A coarser release grid can merge near-matching releases:
    >>> dedouble(df, release_ticks_per_quarter=4)[
    ...     ["original_index", "duplicated_indices"]
    ... ]  # doctest: +NORMALIZE_WHITESPACE
       original_index duplicated_indices
    0               0
    1               1
    2               2                2,3
    3               4
    4               5

    """
    df = df.copy()
    df.attrs["n_undedoubled_notes"] = int((df.type == "note").sum())
    if not match_releases:
        raise NotImplementedError

    # Resolve instrument columns
    if instrument_columns is None:
        resolved_inst_cols = [
            c for c in CANDIDATE_INSTRUMENT_COLUMNS if c in df.columns
        ]
    else:
        resolved_inst_cols = list(instrument_columns)

    release_tpq = (
        release_ticks_per_quarter
        if release_ticks_per_quarter is not None
        else ticks_per_quarter
    )
    if quantize:
        df["quantized_onset"] = (df["onset"] * ticks_per_quarter).round()
        df["quantized_release"] = (df["release"] * release_tpq).round()
        columns = list(columns) + ["quantized_onset", "quantized_release"]
    else:
        columns = list(columns) + ["onset", "release"]

    columns = columns + resolved_inst_cols

    # Group by the columns and get the indices of the rows in each group
    groups = df.groupby(columns).indices

    # Filter out the groups with only one row (i.e., no duplicates)
    duplicate_indices = [indices for indices in groups.values() if len(indices) > 1]

    df["duplicated_indices"] = ""
    to_drop = []
    for indices in duplicate_indices:
        df.loc[indices[0], "duplicated_indices"] = ",".join(str(x) for x in indices)
        to_drop.extend(indices[1:])

    deduplicated = df.drop(to_drop, axis=0)

    if quantize:
        deduplicated = deduplicated.drop(
            ["quantized_onset", "quantized_release"], axis=1
        )

    if "original_index" not in deduplicated.columns:
        deduplicated = deduplicated.reset_index(names="original_index")
    else:
        deduplicated = deduplicated.reset_index(drop=True)

    deduplicated.attrs["n_dedoubled_notes"] = int((deduplicated.type == "note").sum())
    deduplicated.attrs["dedoubled"] = True

    return deduplicated
