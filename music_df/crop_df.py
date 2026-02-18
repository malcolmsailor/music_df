import logging
from typing import Any

import pandas as pd

from music_df.add_feature import infer_barlines
from music_df.utils.search import get_idx_to_item_geq, get_idx_to_item_leq

LOGGER = logging.getLogger(__name__)


def _last_item_before(music_df: pd.DataFrame, i: Any, item_type: str) -> pd.Series:
    # find the last row in music_df.loc[:i] with type item_type:
    # We need to use the `iloc[:-1]` because label-based indexing is inclusive in Pandas
    df_subset = music_df.loc[:i].iloc[:-1]
    try:
        time_sig_i = df_subset[df_subset.type == item_type].index[-1]
    except IndexError:
        raise ValueError(f"No {item_type} before index {i}")
    return music_df.loc[time_sig_i]


def last_bar_before(music_df: pd.DataFrame, i: Any) -> pd.Series:
    return _last_item_before(music_df, i, "bar")


def last_time_signature_before(music_df: pd.DataFrame, i: Any) -> pd.Series:
    """
    Behavior is undefined if `i` is not in `music_df.index`.

    >>> music_df = pd.DataFrame(
    ...     {
    ...         "onset": [0, 4, 8, 12],
    ...         "type": ["time_signature"] * 4,
    ...     },
    ...     index=[str(x) for x in range(4)],
    ... )
    >>> last_time_signature_before(music_df, "2")
    onset                 4
    type     time_signature
    Name: 1, dtype: object
    >>> last_time_signature_before(music_df, "0")
    Traceback (most recent call last):
    ValueError: No time_signature before index 0
    """
    return _last_item_before(music_df, i, "time_signature")
    # # find the last row in music_df.loc[:i] with type time_signature:
    # # We need to use the `iloc[:-1]` because label-based indexing is inclusive in Pandas
    # df_subset = music_df.loc[:i].iloc[:-1]
    # try:
    #     time_sig_i = df_subset[df_subset.type == "time_signature"].index[-1]
    # except IndexError:
    #     raise ValueError(f"No time signature before index {i}")
    # return music_df.loc[time_sig_i]


def crop_df(
    music_df: pd.DataFrame,
    start_i: Any | None = None,
    start_time: float | None = None,
    end_i: Any | None = None,
    end_time: float | None = None,
    infer_barlines_if_no_barlines_found: bool = True,
) -> pd.DataFrame:
    """
    Crop `music_df`, handling time signatures and barlines gracefully.

    Included in the returned dataframe will be the barline and time signature that
    precede the first note. The barline will have its onset preserved, while the
    time signature will be moved to align with the barline if it is not already.

    Args:
        music_df: dataframe. Note that we assume that the dataframe is sorted (see
            music_df.sort_df)
        start_i: index of the first row to include in the cropped dataframe. It is
            assumed that this index points to a note. At most one of `start_i` or
            `start_time` should be provided.
        start_time: onset time of the first note to include in the cropped dataframe.
            At most one of `start_i` or `start_time` should be provided.
        end_i: index of the last row to include in the cropped dataframe. It is
            assumed that this index points to a note. Note that end_i is inclusive (like
            label-based indexing in Pandas). At most one of `end_i` or `end_time` should
            be provided.
        end_time: not implemented.
        infer_barlines_if_no_barlines_found: if True, infer barlines if no barlines are
            found.

    # Examples:

    First define an example dataframe with time signature, note, and bar events:
    >>> music_df = pd.DataFrame(
    ...     {
    ...         "onset": [0.0, 0, 1, 2, 4, 5, 6, 8, 8, 9, 10, 11, 12, 13],
    ...         "release": [
    ...             None,
    ...             4.0,
    ...             4.0,
    ...             3.0,
    ...             8.0,
    ...             8,
    ...             7,
    ...             None,
    ...             11,
    ...             11,
    ...             10.5,
    ...             14,
    ...             14,
    ...             13.5,
    ...         ],
    ...         "type": ["time_signature", "bar", "note", "note", "bar", "note", "note"]
    ...         * 2,
    ...     }
    ... )
    >>> music_df
        onset  release            type
    0     0.0      NaN  time_signature
    1     0.0      4.0             bar
    2     1.0      4.0            note
    3     2.0      3.0            note
    4     4.0      8.0             bar
    5     5.0      8.0            note
    6     6.0      7.0            note
    7     8.0      NaN  time_signature
    8     8.0     11.0             bar
    9     9.0     11.0            note
    10   10.0     10.5            note
    11   11.0     14.0             bar
    12   12.0     14.0            note
    13   13.0     13.5            note


    >>> crop_df(music_df, start_time=2.0)  # doctest: +ELLIPSIS
        onset  release            type
    0     0.0      NaN  time_signature
    1     0.0      4.0             bar
    3     2.0      3.0            note
    4     ...

    >>> crop_df(music_df, start_time=4)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
       onset release            type
    0    4.0     NaN  time_signature
    4    4.0     8.0             bar
    5    5.0     8.0            note
    6    ...
    >>> crop_df(music_df, start_time=10.9)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
       onset release            type
    7   11.0     NaN  time_signature
    11  11.0    14.0             bar
    12  12.0    14.0            note
    13  ...
    >>> crop_df(music_df, start_i=12)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
       onset release            type
    7   11.0     NaN  time_signature
    11  11.0    14.0             bar
    12  12.0    14.0            note
    13  13.0    13.5            note

    Note that end_i is inclusive (like label-based indexing in Pandas):
    >>> crop_df(music_df, end_i=3)  # doctest: +NORMALIZE_WHITESPACE
       onset  release            type
    0    0.0      NaN  time_signature
    1    0.0      4.0             bar
    2    1.0      4.0            note
    3    2.0      3.0            note

    end_time is not yet implemented (releases being unsorted makes it a wee bit
    complicated and I don't yet need it)
    >>> crop_df(music_df, start_time=6, end_time=10.5)
    Traceback (most recent call last):
    NotImplementedError

    >>> crop_df(music_df, start_time=6, end_i=10)  # doctest: +NORMALIZE_WHITESPACE
       onset release            type
    0    4.0     NaN  time_signature
    4    4.0     8.0             bar
    6    6.0     7.0            note
    7    8.0     NaN  time_signature
    8    8.0    11.0             bar
    9    9.0    11.0            note
    10  10.0    10.5            note
    """
    if all(x is None for x in (start_i, start_time, end_i, end_time)):
        LOGGER.warning("Nothing to crop, returning dataframe unchanged")
        return music_df
    if start_i is not None and start_time is not None:
        raise ValueError
    if end_i is not None and end_time is not None:
        raise ValueError

    if start_time is not None:
        notes_df = music_df[music_df.type == "note"]
        start_loc_i = get_idx_to_item_geq(notes_df.onset.values, start_time)
        start_i = notes_df.index[start_loc_i]
    if start_i is not None:
        start_i = int(start_i)
        # TODO: (Malcolm 2023-09-29) what to do in case of no time signature or bar?
        prev_time_sig = last_time_signature_before(music_df, start_i)
        if infer_barlines_if_no_barlines_found and "bar" not in music_df["type"].values:
            music_df = infer_barlines(music_df, keep_old_index=True)
            start_i = music_df[music_df["index"] == start_i].index[0]
            if end_i is not None:
                end_i = music_df[music_df["index"] == end_i].index[0]
            music_df = music_df.drop("index", axis=1)

        prev_bar = last_bar_before(music_df, start_i)
        if prev_time_sig.onset != prev_bar.onset:
            prev_time_sig = prev_time_sig.copy()
            prev_time_sig["onset"] = prev_bar.onset
        # Concatenate prev_time_sig series, prev_bar series, and music_df dataframe:
        music_df = pd.concat(
            [
                prev_time_sig.to_frame().T.astype(music_df.dtypes),
                prev_bar.to_frame().T.astype(music_df.dtypes),
                music_df.loc[start_i:],
            ]
        )

    if end_time is not None:
        raise NotImplementedError
        # We can not use the efficient search because releases may not be in sorted
        # order.
        # However: the below will potentially include releases that go beyond the end
        #   time

        # Get last value of df.release <= end_time
        notes_df = music_df[music_df.type == "note"]
        end_i = notes_df.release[notes_df.release <= end_time].index[-1]
    if end_i is not None:
        end_i = int(end_i)
        music_df = music_df.loc[:end_i]
    return music_df


def get_context_bars(
    music_df: pd.DataFrame,
    target_onset: float,
    n_before: int = 1,
    n_after: int = 1,
) -> pd.DataFrame:
    """
    Extract a window of bars around a target onset.

    Returns a slice of ``music_df`` spanning ``n_before`` bars before and
    ``n_after`` bars after the bar that contains ``target_onset``, with
    barline and time-signature context prepended (matching ``crop_df``
    conventions).

    Args:
        music_df: sorted music DataFrame (see ``music_df.sort_df``).
        target_onset: the onset around which to center the window.
        n_before: number of bars to include before the target bar.
        n_after: number of bars to include after the target bar.

    Raises:
        ValueError: if no bar rows exist and none can be inferred.

    >>> music_df = pd.DataFrame(
    ...     {
    ...         "onset": [0.0, 0, 1, 2, 4, 5, 6, 8, 8, 9, 10, 11, 12, 13],
    ...         "release": [
    ...             None,
    ...             4.0,
    ...             4.0,
    ...             3.0,
    ...             8.0,
    ...             8,
    ...             7,
    ...             None,
    ...             11,
    ...             11,
    ...             10.5,
    ...             14,
    ...             14,
    ...             13.5,
    ...         ],
    ...         "type": ["time_signature", "bar", "note", "note", "bar", "note", "note"]
    ...         * 2,
    ...     }
    ... )

    Middle bar (onset 4) with 1 bar before and after:
    >>> get_context_bars(music_df, 5.0, n_before=1, n_after=1)
        onset  release            type
    0     0.0      NaN  time_signature
    1     0.0      4.0             bar
    2     1.0      4.0            note
    3     2.0      3.0            note
    4     4.0      8.0             bar
    5     5.0      8.0            note
    6     6.0      7.0            note
    7     8.0      NaN  time_signature
    8     8.0     11.0             bar
    9     9.0     11.0            note
    10   10.0     10.5            note

    Target at first bar with large n_before (clamped):
    >>> get_context_bars(music_df, 1.0, n_before=10, n_after=0)
       onset  release            type
    0    0.0      NaN  time_signature
    1    0.0      4.0             bar
    2    1.0      4.0            note
    3    2.0      3.0            note

    Target at last bar with large n_after (clamped):
    >>> get_context_bars(music_df, 12.0, n_before=0, n_after=10)
        onset  release            type
    7    11.0      NaN  time_signature
    11   11.0     14.0             bar
    12   12.0     14.0            note
    13   13.0     13.5            note

    Single bar (n_before=0, n_after=0):
    >>> get_context_bars(music_df, 5.0, n_before=0, n_after=0)
       onset  release            type
    0    4.0      NaN  time_signature
    4    4.0      8.0             bar
    5    5.0      8.0            note
    6    6.0      7.0            note

    Target exactly at bar boundary:
    >>> get_context_bars(music_df, 8.0, n_before=0, n_after=1)
        onset  release            type
    7     8.0      NaN  time_signature
    8     8.0     11.0             bar
    9     9.0     11.0            note
    10   10.0     10.5            note
    11   11.0     14.0             bar
    12   12.0     14.0            note
    13   13.0     13.5            note
    """
    if "bar" not in music_df["type"].values:
        music_df = infer_barlines(music_df)

    bar_rows = music_df[music_df["type"] == "bar"]
    if bar_rows.empty:
        raise ValueError("No bar rows found in music_df")
    bar_onsets = bar_rows["onset"].values

    target_bar_pos = get_idx_to_item_leq(
        bar_onsets, target_onset, return_first_if_larger=True
    )

    start_bar_pos = max(0, target_bar_pos - n_before)
    end_bar_pos = min(len(bar_onsets) - 1, target_bar_pos + n_after)

    start_df_idx = bar_rows.index[start_bar_pos]

    if end_bar_pos + 1 < len(bar_onsets):
        next_bar_onset = bar_onsets[end_bar_pos + 1]
        next_bar_loc = get_idx_to_item_geq(music_df.onset.values, next_bar_onset)
        end_df_idx = music_df.index[next_bar_loc - 1]
    else:
        end_df_idx = music_df.index[-1]

    result = music_df.loc[start_df_idx:end_df_idx]

    first_bar_onset = bar_onsets[start_bar_pos]
    try:
        prev_time_sig = last_time_signature_before(music_df, start_df_idx)
        if prev_time_sig.onset != first_bar_onset:
            prev_time_sig = prev_time_sig.copy()
            prev_time_sig["onset"] = first_bar_onset
        result = pd.concat(
            [prev_time_sig.to_frame().T.astype(result.dtypes), result]
        )
    except ValueError:
        pass

    return result
