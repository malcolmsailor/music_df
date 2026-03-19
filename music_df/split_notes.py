"""
Functions for splitting notes.
"""

import io  # for doctest # noqa: F401

import numpy as np
import pandas as pd

from music_df.add_feature import add_bar_durs
from music_df.sort_df import sort_df
from music_df.transforms import transform


@transform
def split_notes_at_barlines(
    df: pd.DataFrame,
    min_overhang_dur: float | None = None,
    clear_on_split: list[str] | None = None,
    tie_split_notes: bool = True,
):
    """
    >>> csv_table = '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,4.0
    ... note,64,0.0,4.001
    ... note,67,0.0,12.0
    ... bar,,4.0,8.0
    ... note,72,7.999,12.0
    ... bar,,8.0,12.0
    ... note,76,9.0,9.001
    ... bar,,12.0,16.0
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df["tie_to_next"] = False
    >>> df["tie_to_prev"] = False
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

    >>> split_notes_at_barlines(df, tie_split_notes=False)
        type  pitch   onset  release  tie_to_next  tie_to_prev
    0    bar    NaN   0.000    4.000        False        False
    1   note   60.0   0.000    4.000        False        False
    2   note   64.0   0.000    4.000        False        False
    3   note   67.0   0.000    4.000        False        False
    4    bar    NaN   4.000    8.000        False        False
    5   note   64.0   4.000    4.001        False        False
    6   note   67.0   4.000    8.000        False        False
    7   note   72.0   7.999    8.000        False        False
    8    bar    NaN   8.000   12.000        False        False
    9   note   67.0   8.000   12.000        False        False
    10  note   72.0   8.000   12.000        False        False
    11  note   76.0   9.000    9.001        False        False
    12   bar    NaN  12.000   16.000        False        False
    """
    if df.loc[df.type == "bar", "release"].isna().any():
        df = add_bar_durs(df)
    if tie_split_notes:
        if "tie_to_next" not in df.columns:
            df = df.copy()
            df["tie_to_next"] = False
        if "tie_to_prev" not in df.columns:
            df = df.copy()
            df["tie_to_prev"] = False

    note_mask = df["type"].values == "note"
    non_note_df = df[~note_mask]
    notes_df = df[note_mask]

    if len(notes_df) == 0:
        out_df = df.copy()
        out_df.attrs = df.attrs.copy()
        return sort_df(out_df, force=True)

    # Use both bar onsets and releases as split boundaries so notes
    # extending past the last bar's release are handled correctly
    bars_df = df[df["type"] == "bar"]
    bar_boundaries = np.unique(np.concatenate([
        bars_df["onset"].values,
        bars_df["release"].dropna().values,
    ]))

    if len(bar_boundaries) == 0:
        out_df = df.copy()
        out_df.attrs = df.attrs.copy()
        return sort_df(out_df, force=True)

    onsets = notes_df["onset"].values
    releases = notes_df["release"].values

    # For each note, find bar boundaries strictly between onset and release
    first_boundary_idx = np.searchsorted(bar_boundaries, onsets, side="right")
    last_boundary_idx = np.searchsorted(bar_boundaries, releases, side="left")
    n_splits = np.maximum(last_boundary_idx - first_boundary_idx, 0)
    n_fragments = n_splits + 1

    if not (n_splits > 0).any():
        out_df = df.copy()
        out_df.attrs = df.attrs.copy()
        return sort_df(out_df, force=True)

    n_notes = len(notes_df)
    total_fragments = int(n_fragments.sum())

    # Expand note rows: repeat each note's index by its fragment count
    repeat_counts = n_fragments.astype(np.intp)
    note_indices = np.repeat(np.arange(n_notes), repeat_counts)

    # Fragment index within each note (0, 1, ..., n_fragments[i]-1)
    offsets = np.zeros(total_fragments, dtype=np.intp)
    positions = np.cumsum(repeat_counts)
    offsets[positions[:-1]] = repeat_counts[:-1]
    # After cumsum and subtract, fragment_idx[j] = position within its note
    fragment_idx = np.arange(total_fragments) - np.repeat(
        positions - repeat_counts, repeat_counts
    )

    # Compute onset/release for each fragment
    exp_onsets = onsets[note_indices].copy()
    exp_releases = releases[note_indices].copy()
    exp_first_bi = first_boundary_idx[note_indices]
    exp_n_frags = repeat_counts[note_indices]

    is_first = fragment_idx == 0
    is_last = fragment_idx == exp_n_frags - 1
    was_split = exp_n_frags > 1

    # Non-first fragments: onset = bar_boundaries[first_boundary_idx + frag_idx - 1]
    non_first = ~is_first
    exp_onsets[non_first] = bar_boundaries[
        exp_first_bi[non_first] + fragment_idx[non_first] - 1
    ]

    # Non-last fragments: release = bar_boundaries[first_boundary_idx + frag_idx]
    non_last = ~is_last
    exp_releases[non_last] = bar_boundaries[
        exp_first_bi[non_last] + fragment_idx[non_last]
    ]

    # Build the expanded notes DataFrame
    expanded = notes_df.iloc[note_indices].copy()
    expanded = expanded.reset_index(drop=True)
    expanded["onset"] = exp_onsets
    expanded["release"] = exp_releases

    # Handle ties
    if tie_split_notes:
        tie_next = expanded["tie_to_next"].values.copy()
        tie_prev = expanded["tie_to_prev"].values.copy()

        # Non-last fragments of split notes: tie_to_next = True
        tie_next[non_last & was_split] = True
        # Non-first fragments of split notes: tie_to_prev = True
        tie_prev[non_first & was_split] = True

        expanded["tie_to_next"] = tie_next
        expanded["tie_to_prev"] = tie_prev

    # Handle clear_on_split: clear specified columns on non-first fragments
    if clear_on_split:
        for col in clear_on_split:
            if col in expanded.columns:
                expanded.loc[non_first & was_split, col] = ""

    # Handle min_overhang_dur: filter out short fragments
    if min_overhang_dur is not None:
        durations = exp_releases - exp_onsets
        keep = durations >= min_overhang_dur
        # Unsplit notes are always kept
        keep = keep | ~was_split

        if tie_split_notes:
            # Adjust ties: if a fragment is removed, adjacent fragments
            # shouldn't point to it.
            # For each note group, tie_to_next should be True only if the
            # NEXT kept fragment exists; tie_to_prev only if the PREVIOUS
            # kept fragment exists.
            tie_next = expanded["tie_to_next"].values.copy()
            tie_prev = expanded["tie_to_prev"].values.copy()

            # Process per-note: for notes that had splits and have removals
            notes_with_removals = np.unique(note_indices[~keep & was_split])
            for ni in notes_with_removals:
                mask = note_indices == ni
                frag_keep = keep[mask]
                n_kept = frag_keep.sum()
                frag_positions = np.where(mask)[0]

                if n_kept <= 1:
                    # Single or no fragments left: no ties
                    tie_next[frag_positions] = False
                    tie_prev[frag_positions] = False
                    # Restore original tie values for the surviving fragment
                    if n_kept == 1:
                        kept_pos = frag_positions[frag_keep][0]
                        kept_frag_idx = fragment_idx[kept_pos]
                        if kept_frag_idx == 0:
                            tie_next[kept_pos] = False
                        elif kept_frag_idx == exp_n_frags[kept_pos] - 1:
                            tie_prev[kept_pos] = False
                else:
                    kept_positions = frag_positions[frag_keep]
                    # First kept: no tie_to_prev (unless it was the original
                    # first fragment with an existing tie)
                    if fragment_idx[kept_positions[0]] != 0:
                        tie_prev[kept_positions[0]] = False
                    # Last kept: no tie_to_next (unless it was the original
                    # last fragment with an existing tie)
                    if fragment_idx[kept_positions[-1]] != exp_n_frags[kept_positions[-1]] - 1:
                        tie_next[kept_positions[-1]] = False

            expanded["tie_to_next"] = tie_next
            expanded["tie_to_prev"] = tie_prev

        expanded = expanded[keep].reset_index(drop=True)

    # Combine non-notes and expanded notes
    out_df = pd.concat([non_note_df, expanded], ignore_index=True)
    out_df.attrs = df.attrs.copy()
    out_df = sort_df(out_df, force=True)
    return out_df


@transform
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

    out = pd.DataFrame(new_rows).reset_index(drop=True)
    out.attrs = df.attrs.copy()
    return out
