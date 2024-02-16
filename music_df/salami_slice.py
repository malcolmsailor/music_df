import numpy as np
import pandas as pd

from music_df.quantize_df import quantize_df
from music_df.sort_df import sort_df
from music_df.sync_df import get_unique_from_array_by_df


def appears_salami_sliced(df: pd.DataFrame) -> bool:
    """
    Returns True if every unique onset has a unique release associated with it, and vice
    versa.

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

    This doesn't actually check for overlapping notes, however.
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
    first_index = grouped.apply(lambda x: x.index[0])
    return df.loc[first_index]


def salami_slice(df: pd.DataFrame) -> pd.DataFrame:
    # TODO implement a "minimum tolerance" so that onsets/releases don't
    #   have to be *exactly* simultaneous
    # any zero-length notes will be omitted.
    # Given that all onsets/releases will be homophonic after running
    #   this function, there would be a more efficient way of storing notes
    #   than storing each one individually, but then we would have to rewrite
    #   repr functions for the output.
    if len(df) == 0:
        return df.copy()
    moments = sorted(
        set(df[df.type == "note"].onset) | set(df[df.type == "note"].release)
    )
    moment_iter = enumerate(moments)
    moment_i, moment = next(moment_iter)
    out = []
    for _, note in df.iterrows():
        if note.type != "note":
            out.append(note.copy())
            continue
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
    sort_df(new_df, inplace=True)
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
