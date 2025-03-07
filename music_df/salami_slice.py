"""
This module contains functions for salami-slicing music dataframes.

A salami-sliced score is a score where, at each timestep with one or more onsets or
releases, we split any ongoing notes into two, in order to obtain a purely homophonic
rhythmic texture in which all onsets and releases are synchronized across all parts.
"""

from io import StringIO  # Used by doctests

import numpy as np
import pandas as pd

from music_df.quantize_df import quantize_df
from music_df.sort_df import sort_df


def appears_salami_sliced(df: pd.DataFrame) -> bool:
    """
    Evaluates heuristically whether a dataframe has been salami sliced.

    Returns True if every unique onset has a unique release associated with it, and vice
    versa.

    This doesn't actually check for overlapping notes, however.

    >>> df = pd.DataFrame(
    ...     {"pitch": [60, 61], "onset": [0.0, 0.0], "release": [0.5, 1.0]}
    ... )
    >>> appears_salami_sliced(df)
    False
    >>> df = pd.DataFrame(
    ...     {"pitch": [60, 61], "onset": [0.0, 0.0], "release": [1.0, 1.0]}
    ... )
    >>> appears_salami_sliced(df)
    True
    >>> df = pd.DataFrame(
    ...     {"pitch": [60, 61], "onset": [0.0, 1.0], "release": [2.0, 2.0]}
    ... )
    >>> appears_salami_sliced(df)
    False
    >>> df = pd.DataFrame(
    ...     {"pitch": [60, 61], "onset": [0.0, 1.0], "release": [1.0, 2.0]}
    ... )
    >>> appears_salami_sliced(df)
    True

    Overlapping notes are not detected:

    >>> df = pd.DataFrame(
    ...     {"pitch": [60, 61], "onset": [0.0, 1.0], "release": [4.0, 2.0]}
    ... )
    >>> appears_salami_sliced(df)
    True
    """
    if "type" in df.columns:
        df = df[df.type == "note"]
    for x, y in (("onset", "release"), ("release", "onset")):
        grouped = df.groupby(x)
        for _, group_df in grouped:
            if len(group_df[y].unique()) > 1:
                return False

    return True


def get_unique_salami_slices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with only the first row in each slice.

    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 61, 62],
    ...         "onset": [0.0, 1.0, 1.0],
    ...         "release": [1.0, 2.0, 2.0],
    ...     }
    ... )
    >>> get_unique_salami_slices(df)
       pitch  onset  release
    0     60    0.0      1.0
    1     61    1.0      2.0
    """
    assert appears_salami_sliced(df)
    grouped = df.groupby("onset")
    first_index = grouped.apply(lambda x: x.index[0], include_groups=False)
    return df.loc[first_index]


def salami_slice(
    df: pd.DataFrame,
    include_slice_ids: bool = True,
    label_original_note_ids: bool = False,
) -> pd.DataFrame:
    """
    Returns a salami-sliced version of the dataframe.

    A salami-sliced score is a score where, at each timestep with one or more onsets or
    releases, we split any ongoing notes into two, in order to obtain a purely
    homophonic rhythmic texture in which all onsets and releases are synchronized across
    all parts.

    It's probably best to quantize the dataframe before running this function so that
    very nearly simultaneous onsets/releases are synchronized.

    Any zero-length notes will be omitted.
    """
    # Given that all onsets/releases will be homophonic after running
    #   this function, there would be a more efficient way of storing notes
    #   than storing each one individually, but then we would have to rewrite
    #   repr functions for the output.
    if len(df) == 0:
        out = df.copy()
        out.attrs["salami_sliced"] = True
        out.attrs["n_salami_sliced_notes"] = 0
        out.attrs["n_unsalami_sliced_notes"] = 0
        return out

    moments = sorted(
        set(df[df.type == "note"].onset) | set(df[df.type == "note"].release)
    )
    moment_iter = enumerate(moments)
    moment_i, moment = next(moment_iter)
    out = []
    for row_id, note in df.iterrows():
        if note.type != "note":
            out.append(note.copy())
            continue

        if label_original_note_ids:
            note["original_note_id"] = row_id

        onset_i = 0
        while note.onset > moment:
            moment_i, moment = next(moment_iter)
            onset_i += 1
        onset = note.onset
        release_i = moment_i + 1
        while release_i < len(moments) and moments[release_i] <= note.release:
            new_note = note.copy()
            new_note.onset = onset
            new_note.release = onset = moments[release_i]
            out.append(new_note)
            release_i += 1
    new_df = pd.DataFrame(out)
    new_df.attrs = df.attrs.copy()
    sort_df(new_df, inplace=True)

    new_df.attrs["salami_sliced"] = True
    new_df.attrs["n_salami_sliced_notes"] = int((new_df.type == "note").sum())
    new_df.attrs["n_unsalami_sliced_notes"] = int((df.type == "note").sum())
    if include_slice_ids:
        new_df = add_slice_ids(new_df, check_salami_sliced=False)
        new_df = add_distinct_slice_ids(new_df, check_salami_sliced=False)
    return new_df


def slice_into_uniform_steps(
    df: pd.DataFrame, step_dur: float, quantize_tpq: None | int = None
) -> pd.DataFrame:
    """
    >>> df = pd.DataFrame(
    ...     {
    ...         "type": ["bar", "note", "note"],
    ...         "pitch": [float("nan"), 60, 61],
    ...         "onset": [0.0, 0.0, 1.0],
    ...         "release": [4.0, 1.0, 2.0],
    ...     }
    ... )
    >>> slice_into_uniform_steps(df, step_dur=0.5, quantize_tpq=4)
       type  pitch  onset  release
    0   bar    NaN    0.0      4.0
    1  note   60.0    0.0      1.0
    1  note   60.0    0.0      0.5
    2  note   61.0    1.0      2.0
    2  note   61.0    1.0      1.5
    >>> df = pd.DataFrame(
    ...     {
    ...         "type": ["bar", "note", "note"],
    ...         "pitch": [float("nan"), 60, 61],
    ...         "onset": [0.0, 0.1, 0.9],
    ...         "release": [4.0, 0.9, 1.9],
    ...     }
    ... )
    >>> slice_into_uniform_steps(df, step_dur=0.5, quantize_tpq=4)
       type  pitch  onset  release
    0   bar    NaN    0.0      4.0
    1  note   60.0    0.0      1.0
    1  note   60.0    0.0      0.5
    2  note   61.0    1.0      2.0
    2  note   61.0    1.0      1.5

    Quantization that doesn't divide step_dur evenly isn't supported yet:
    >>> slice_into_uniform_steps(df, step_dur=0.5, quantize_tpq=5)
    Traceback (most recent call last):
    NotImplementedError

    Similarly, unquantized dataframes are not yet supported:
    >>> slice_into_uniform_steps(df, step_dur=0.5, quantize_tpq=None)
    Traceback (most recent call last):
    NotImplementedError

    """
    if quantize_tpq is not None:
        df = quantize_df(df, tpq=quantize_tpq)
    if quantize_tpq is None or step_dur % (1 / quantize_tpq) != 0:
        raise NotImplementedError
    rows = []
    for _, row in df.iterrows():
        if row["type"] != "note":
            rows.append(row)
            continue

        time_range = np.arange(row["onset"], row["release"], step_dur)
        for onset in time_range:
            row_copy = row.copy()
            row["onset"] = onset
            row["release"] = onset + step_dur
            rows.append(row_copy)
    return pd.DataFrame(rows)


def add_slice_ids(df: pd.DataFrame, check_salami_sliced: bool = True):
    """
    Add a unique slice id to each note.

    The slice ids will be increasing integers, starting at 0. Non-note rows are given a
    slice id of -1.

    >>> table = '''
    ...    type  pitch  onset  release
    ... 0   bar    NaN    0.0      4.0
    ... 1  note   60.0    0.0      1.0
    ... 2  note   64.0    0.0      1.0
    ... 3  note   61.0    1.0      2.0
    ... 4  note   65.0    1.0      2.0
    ... 5   bar    NaN    4.0      8.0
    ... '''
    >>> df = pd.read_csv(StringIO(table), sep="\\\\s+")
    >>> add_slice_ids(df)
       type  pitch  onset  release  slice_id
    0   bar    NaN    0.0      4.0        -1
    1  note   60.0    0.0      1.0         0
    2  note   64.0    0.0      1.0         0
    3  note   61.0    1.0      2.0         1
    4  note   65.0    1.0      2.0         1
    5   bar    NaN    4.0      8.0        -1

    """
    if check_salami_sliced:
        assert appears_salami_sliced(df)
    note_mask = df["type"] == "note"
    df["slice_id"] = -1
    df.loc[note_mask, "slice_id"] = pd.factorize(df.loc[note_mask, "onset"])[0]
    return df


def add_distinct_slice_ids(df: pd.DataFrame, check_salami_sliced: bool = True):
    """
    Add "distinct" slice ids to each note.

    A salami slice is "distinct" if its pitch contents differ from those in the
    previous slice. Thus "distinct" salami slices capture changes of pitch in the
    musical content.

    As with `add_slice_ids`, the slice ids will be increasing integers, starting at 0.
    Non-note rows are given a distinct slice id of -1.

    >>> table = '''
    ...    type  pitch  onset  release
    ... 0   bar    NaN    0.0      2.0
    ... 1  note   60.0    0.0      1.0
    ... 2  note   64.0    0.0      1.0
    ... 3  note   60.0    1.0      2.0
    ... 4  note   64.0    1.0      2.0
    ... 5   bar    NaN    2.0      4.0
    ... 6  note   60.0    2.0      3.0
    ... 7  note   65.0    2.0      3.0
    ... '''
    >>> df = pd.read_csv(StringIO(table), sep="\\\\s+")
    >>> add_distinct_slice_ids(df)
       type  pitch  onset  release  distinct_slice_id
    0   bar    NaN    0.0      2.0                 -1
    1  note   60.0    0.0      1.0                  0
    2  note   64.0    0.0      1.0                  0
    3  note   60.0    1.0      2.0                  0
    4  note   64.0    1.0      2.0                  0
    5   bar    NaN    2.0      4.0                 -1
    6  note   60.0    2.0      3.0                  1
    7  note   65.0    2.0      3.0                  1
    """
    if check_salami_sliced:
        assert appears_salami_sliced(df)

    prev_pitches = None
    slice_id = -1

    df["distinct_slice_id"] = -1

    for name, group in df[df["type"] == "note"].groupby("onset"):
        if (these_pitches := set(group["pitch"])) != prev_pitches:
            slice_id += 1
            prev_pitches = these_pitches
        df.loc[group.index, "distinct_slice_id"] = slice_id

    return df


def undo_salami_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inverts the salami slicing operation.

    Requires the `original_note_id` column, which is added by `salami_slice`, in order
    to specify which notes should be merged.
    """
    if "original_note_id" not in df.columns:
        raise ValueError(
            "We require the 'original_note_id' column to undo salami slicing"
        )
    note_mask = df["type"] == "note"
    non_notes = df[~note_mask]
    notes = df[note_mask]
    merged_notes = []
    for note_id, notes in notes.groupby("original_note_id"):
        # TODO remove these assertions when I'm confident it works as expected
        assert notes.iloc[0].onset == notes.onset.min()
        assert notes.iloc[-1].release == notes.release.max()
        merged_note = notes.iloc[0].copy()
        merged_note.release = notes.iloc[-1].release
        merged_notes.append(merged_note)
    out = pd.concat([non_notes, pd.DataFrame(merged_notes)])
    out = sort_df(out, inplace=True)
    out.attrs = df.attrs.copy()
    out.drop(columns=["original_note_id"], inplace=True)

    return out
