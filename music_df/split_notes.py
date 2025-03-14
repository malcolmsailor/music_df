"""
Functions for splitting notes.
"""

import io  # for doctest # noqa: F401

import numpy as np
import pandas as pd

from music_df.add_feature import add_bar_durs
from music_df.sort_df import sort_df


def split_notes_at_barlines(
    df: pd.DataFrame,
    min_overhang_dur: float | None = None,
):
    """
    >>> csv_table = '''
    ... type,pitch,onset,release,tie_to_next,tie_to_prev
    ... bar,,0.0,4.0,,
    ... note,60,0.0,4.0,,
    ... note,64,0.0,4.001,,
    ... note,67,0.0,12.0,,
    ... bar,,4.0,8.0,,
    ... note,72,7.999,12.0,,
    ... bar,,8.0,12.0,,
    ... note,76,9.0,9.001,,
    ... bar,,12.0,16.0,,
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df["tie_to_next"] = df["tie_to_next"].fillna(False)
    >>> df["tie_to_prev"] = df["tie_to_prev"].fillna(False)
    >>> df
       type  pitch   onset  release  tie_to_next  tie_to_prev
    0   bar    NaN   0.000    4.000        False        False
    1  note   60.0   0.000    4.000        False        False
    2  note   64.0   0.000    4.001        False        False
    3  note   67.0   0.000   12.000        False        False
    4   bar    NaN   4.000    8.000        False        False
    5  note   72.0   7.999   12.000        False        False
    6   bar    NaN   8.000   12.000        False        False
    7  note   76.0   9.000    9.001        False        False
    8   bar    NaN  12.000   16.000        False        False

    >>> split_notes_at_barlines(df)
        type  pitch   onset  release  tie_to_next  tie_to_prev
    0    bar    NaN   0.000    4.000        False        False
    1   note   60.0   0.000    4.000        False        False
    2   note   64.0   0.000    4.000         True        False
    3   note   67.0   0.000    4.000         True        False
    4    bar    NaN   4.000    8.000        False        False
    5   note   64.0   4.000    4.001        False         True
    6   note   67.0   4.000    8.000         True         True
    7   note   72.0   7.999    8.000         True        False
    8    bar    NaN   8.000   12.000        False        False
    9   note   67.0   8.000   12.000        False         True
    10  note   72.0   8.000   12.000        False         True
    11  note   76.0   9.000    9.001        False        False
    12   bar    NaN  12.000   16.000        False        False

    >>> split_notes_at_barlines(df, min_overhang_dur=0.025)
        type  pitch  onset  release  tie_to_next  tie_to_prev
    0    bar    NaN    0.0    4.000        False        False
    1   note   60.0    0.0    4.000        False        False
    2   note   64.0    0.0    4.000        False        False
    3   note   67.0    0.0    4.000         True        False
    4    bar    NaN    4.0    8.000        False        False
    5   note   67.0    4.0    8.000         True         True
    6    bar    NaN    8.0   12.000        False        False
    7   note   67.0    8.0   12.000        False         True
    8   note   72.0    8.0   12.000        False        False
    9   note   76.0    9.0    9.001        False        False
    10   bar    NaN   12.0   16.000        False        False

    """
    if df.loc[df.type == "bar", "release"].isna().any():
        df = add_bar_durs(df)
    bars = df[df.type == "bar"].reset_index()
    bars_i = 0
    row_accumulator = []
    for _, row in df.iterrows():
        overhang = False
        if row.type != "note":
            row_accumulator.append(row)
            continue
        else:
            while (bars_i < len(bars) - 1) and (bars.loc[bars_i + 1].onset < row.onset):
                bars_i += 1
            temp_row = row.copy()
            for final_bars_i in range(bars_i, len(bars)):
                bar_release = bars.loc[final_bars_i].release
                if row.release > bar_release:
                    overhang = True
                    truncated_row = temp_row.copy()
                    truncated_row.release = bar_release
                    if (
                        min_overhang_dur is None
                        or truncated_row.release - truncated_row.onset
                        >= min_overhang_dur
                    ):
                        row_accumulator.append(truncated_row)
                        temp_row.tie_to_prev = True

                    temp_row.onset = bar_release
                    if (
                        min_overhang_dur is None
                        or temp_row.release - temp_row.onset >= min_overhang_dur
                    ):
                        truncated_row.tie_to_next = True
                else:
                    break
            if (
                (not overhang)
                or (min_overhang_dur is None)
                or (temp_row.release - temp_row.onset >= min_overhang_dur)
            ):
                row_accumulator.append(temp_row)

    out_df = pd.DataFrame(row_accumulator)
    out_df = sort_df(out_df)
    return out_df


def subdivide_notes(
    df: pd.DataFrame, grid_size, onset_col="onset", release_col="release"
):
    """
    Subdivides notes into smaller intervals of size grid_size.

    Parameters:
        df: pandas DataFrame with onset and release times
        grid_size: size of subdivisions
        onset_col: name of onset column
        release_col: name of release column

    Returns:
        DataFrame with subdivided intervals


    >>> csv_table = '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,4.0
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> subdivide_notes(df, 1.0)
       type  pitch  onset  release
    0   bar    NaN    0.0      4.0
    1  note   60.0    0.0      1.0
    2  note   60.0    1.0      2.0
    3  note   60.0    2.0      3.0
    4  note   60.0    3.0      4.0
    """
    new_rows = []

    for _, row in df.iterrows():
        if row["type"] != "note":
            new_rows.append(row)
            continue

        # Get start and end times
        start = row[onset_col]
        end = row[release_col]

        # Find the first interval boundary after start
        first_boundary = np.ceil(start / grid_size) * grid_size

        # Generate all interval boundaries
        boundaries = np.arange(first_boundary, end, grid_size)

        # Create subdivided intervals
        if len(boundaries) == 0:
            # Case where interval is smaller than grid_size
            new_rows.append(pd.Series({**row, onset_col: start, release_col: end}))
        else:
            # First interval (from start to first boundary)
            if start < boundaries[0]:
                new_rows.append(
                    pd.Series(
                        {**row, onset_col: start, release_col: min(boundaries[0], end)}
                    )
                )

            # Middle intervals
            for i in range(len(boundaries)):
                if boundaries[i] >= end:
                    break

                new_rows.append(
                    pd.Series(
                        {
                            **row,
                            onset_col: boundaries[i],
                            release_col: min(boundaries[i] + grid_size, end),
                        },
                    )
                )

    return pd.DataFrame(new_rows).reset_index(drop=True)
