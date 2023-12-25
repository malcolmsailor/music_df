from fractions import Fraction
from math import isnan
from typing import Iterable

import pandas as pd

from ast import literal_eval
from music_df.add_feature import add_bar_durs
from music_df.utils.search import get_index_to_item_leq


def _to_dict_if_necessary(d: str | dict) -> dict:
    if isinstance(d, str):
        return literal_eval(d)
    return d


def get_time_sig_dur(time_sig: dict[str, int] | str):
    time_sig = _to_dict_if_necessary(time_sig)
    return time_sig["numerator"] * 4 / time_sig["denominator"]


def appears_to_have_pickup_measure(music_df: pd.DataFrame) -> bool:
    """

    Caches a "appears_to_have_pickup_measure" boolean attribute in music_df.attrs.

    If

    If there is a time signature, and there is at least one bar, we return whether
    the first bar is shorter than the time signature:

    >>> df = pd.DataFrame(
    ...     [
    ...         {
    ...             "type": "time_signature",
    ...             "onset": 0.0,
    ...             "release": float("nan"),
    ...             "other": {"numerator": 4, "denominator": 4},
    ...         },
    ...         {"type": "bar", "onset": 0.0, "release": 4.0},
    ...         {"type": "bar", "onset": 4.0, "release": 8.0},
    ...     ]
    ... )
    >>> appears_to_have_pickup_measure(df)
    False

    >>> df = pd.DataFrame(
    ...     [
    ...         {
    ...             "type": "time_signature",
    ...             "onset": 0.0,
    ...             "release": float("nan"),
    ...             "other": {"numerator": 4, "denominator": 4},
    ...         },
    ...         {"type": "bar", "onset": 0.0, "release": 2.0},
    ...         {"type": "bar", "onset": 2.0, "release": 6.0},
    ...     ]
    ... )
    >>> appears_to_have_pickup_measure(df)
    True

    If there is no time signature, but there are at least two bars, return whether
    the first bar is shorter than the second bar.

    >>> df = pd.DataFrame(
    ...     [
    ...         {"type": "bar", "onset": 0.0, "release": 4.0},
    ...         {"type": "bar", "onset": 4.0, "release": 8.0},
    ...     ]
    ... )
    >>> appears_to_have_pickup_measure(df)
    False

    >>> df = pd.DataFrame(
    ...     [
    ...         {"type": "bar", "onset": 0.0, "release": 4.0},
    ...         {"type": "bar", "onset": 4.0, "release": 12.0},
    ...     ]
    ... )
    >>> appears_to_have_pickup_measure(df)
    True

    Otherwise, return False.

    >>> df = pd.DataFrame([{"type": "bar", "onset": 0.0, "release": 4.0}])
    >>> appears_to_have_pickup_measure(df)
    False
    >>> df = pd.DataFrame()
    >>> appears_to_have_pickup_measure(df)
    False
    >>> df = pd.DataFrame(
    ...     [
    ...         {
    ...             "type": "time_signature",
    ...             "onset": 0.0,
    ...             "release": float("nan"),
    ...             "other": {"numerator": 4, "denominator": 4},
    ...         }
    ...     ]
    ... )
    >>> appears_to_have_pickup_measure(df)
    False
    """
    if "appears_to_have_pickup_measure" in music_df.attrs:
        return music_df.attrs["appears_to_have_pickup_measure"]

    def _sub(music_df):
        if not len(music_df):
            return False

        bar_mask = music_df.type == "bar"
        if bar_mask.sum() < 1:
            return False

        if isnan(music_df[bar_mask].iloc[0].release):
            music_df = add_bar_durs(music_df)
        first_bar = music_df[bar_mask].iloc[0]
        first_bar_dur = first_bar.release - first_bar.onset

        time_sig_mask = music_df.type == "time_signature"
        if (
            not time_sig_mask.any()
            or (first_time_sig := music_df[time_sig_mask].iloc[0]).onset != 0
        ):
            # Return whether first measure is shorter than second measure
            if bar_mask.sum() < 2:
                return False

            if isnan(music_df[bar_mask].iloc[1].release):
                music_df = add_bar_durs(music_df)

            second_bar = music_df[bar_mask].iloc[1]
            second_bar_dur = second_bar.release - second_bar.onset
            return first_bar_dur < second_bar_dur

        time_sig_dur = get_time_sig_dur(first_time_sig.other)
        return time_sig_dur > first_bar_dur

    out = _sub(music_df)
    music_df.attrs["appears_to_have_pickup_measure"] = out
    return out


def time_to_bar_number_and_offset(
    music_df: pd.DataFrame, x: float | Fraction | int
) -> tuple[int, float]:
    """
    If there doesn't appear to be a pickup measure, the first measure is `1`.

    >>> df = pd.DataFrame(
    ...     [
    ...         {
    ...             "type": "time_signature",
    ...             "onset": 0.0,
    ...             "release": float("nan"),
    ...             "other": {"numerator": 4, "denominator": 4},
    ...         },
    ...         {"type": "bar", "onset": 0.0, "release": 4.0},
    ...         {"type": "bar", "onset": 4.0, "release": 8.0},
    ...     ]
    ... )
    >>> time_to_bar_number_and_offset(df, 0.0)
    (1, 0.0)
    >>> time_to_bar_number_and_offset(df, 3.0)
    (1, 3.0)
    >>> time_to_bar_number_and_offset(df, 4.0)
    (2, 0.0)

    If there does appear to be a pickup measure, the first measure is numbered 0. (So
    the first full measure is 1.)
    >>> df = pd.DataFrame(
    ...     [
    ...         {
    ...             "type": "time_signature",
    ...             "onset": 0.0,
    ...             "release": float("nan"),
    ...             "other": {"numerator": 4, "denominator": 2},
    ...         },
    ...         {"type": "bar", "onset": 0.0, "release": 4.0},
    ...         {"type": "bar", "onset": 4.0, "release": 12.0},
    ...     ]
    ... )
    >>> time_to_bar_number_and_offset(df, 0.0)
    (0, 0.0)
    >>> time_to_bar_number_and_offset(df, 3.0)
    (0, 3.0)
    >>> time_to_bar_number_and_offset(df, 4.0)
    (1, 0.0)
    """
    bar_mask = music_df.type == "bar"
    assert bar_mask.any()

    bars = music_df[bar_mask].reset_index(drop=True)
    bar_i = get_index_to_item_leq(bars.onset, val=x)

    assert bar_i % 1 == 0  # type:ignore
    bar_i = int(bar_i)  # type:ignore

    bar_number = bar_i if appears_to_have_pickup_measure(music_df) else bar_i + 1
    bar_onset = float(bars.loc[bar_i, "onset"])  # type:ignore
    offset = float(x - bar_onset)
    return bar_number, offset


def merge_contiguous_durations(
    durs: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    >>> merge_contiguous_durations([])
    []
    >>> merge_contiguous_durations([(0.0, 4.0)])
    [(0.0, 4.0)]
    >>> merge_contiguous_durations([(0.0, 4.0), (4.0, 8.0)])
    [(0.0, 8.0)]
    >>> merge_contiguous_durations([(0.0, 4.0), (5.0, 8.0)])
    [(0.0, 4.0), (5.0, 8.0)]

    Overlapping durations are returned as-is.
    >>> merge_contiguous_durations([(0.0, 4.0), (3.0, 8.0)])
    [(0.0, 4.0), (3.0, 8.0)]

    >>> merge_contiguous_durations([(0.0, 4.0), (5.0, 8.0), (8.0, 9.0), (9.01, 11.0)])
    [(0.0, 4.0), (5.0, 9.0), (9.01, 11.0)]
    """
    out = []
    prev_onset, prev_release = None, None
    for onset, release in durs:
        if prev_release is None:
            prev_onset = onset
        elif prev_release is not None and onset != prev_release:
            out.append((prev_onset, prev_release))
            prev_onset = onset
        prev_release = release
    if prev_release is not None:
        out.append((prev_onset, prev_release))
    return out
