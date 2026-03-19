from typing import Literal, get_args

import numpy as np
import pandas as pd

from music_df.sort_df import sort_df
from music_df.transforms import transform

ZeroDurAction = Literal["remove", "drop", "min_dur", "preserve"]


def _quantize_diff(before_df, after_df):
    """Quantization repositions notes on a grid; the naive set-diff overcounts
    because it sees every repositioned note as removed+added. Only zero-dur
    drops are true removals.
    """
    dropped = after_df.attrs.get("_dropped_note_originals", [])
    if not dropped:
        return set(), set()

    before_notes = before_df[before_df["type"] == "note"]
    removed = set()
    used_indices = set()
    for orig_onset, orig_release, pitch in dropped:
        candidates = before_notes[
            (before_notes["pitch"] == pitch)
            & (~before_notes.index.isin(used_indices))
        ]
        if candidates.empty:
            continue
        best_idx = (candidates["onset"] - orig_onset).abs().idxmin()
        row = candidates.loc[best_idx]
        removed.add((row["onset"], row["release"], row["pitch"]))
        used_indices.add(best_idx)

    return removed, set()


@transform(diff_func=_quantize_diff)
def quantize_df(
    df,
    tpq: int = 4,
    ticks_out: bool = False,
    zero_dur_action: ZeroDurAction = "min_dur",
) -> pd.DataFrame:
    """
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 61, 62, 63],
    ...         "onset": [-0.01, 1.01, 1.95, 2.9],
    ...         "release": [0.99, 2.03, 3.0, 3.97],
    ...     }
    ... )
    >>> df
       pitch  onset  release
    0     60  -0.01     0.99
    1     61   1.01     2.03
    2     62   1.95     3.00
    3     63   2.90     3.97

    There may be a negative zero in the output:
    >>> quantize_df(df, tpq=4)
       pitch  onset  release
    0     60   -0.0      1.0
    1     61    1.0      2.0
    2     62    2.0      3.0
    3     63    3.0      4.0

    >>> quantize_df(df, tpq=16)
       pitch   onset  release
    0     60 -0.0000      1.0
    1     61  1.0000      2.0
    2     62  1.9375      3.0
    3     63  2.8750      4.0

    >>> quantize_df(df, tpq=16, ticks_out=True)
       pitch  onset  release
    0     60      0       16
    1     61     16       32
    2     62     31       48
    3     63     46       64

    Note that by default, notes that would be rounded to have zero length
    are given the minimum length.
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 61, 62],
    ...         "onset": [0.0, 0.5, 1.0],
    ...         "release": [0.4, 1.0, 2.0],
    ...     }
    ... )
    >>> quantize_df(df, tpq=1)
       pitch  onset  release
    0     60    0.0      1.0
    1     61    0.0      1.0
    2     62    1.0      2.0

    To preserve zero-dur notes, pass `zero_dur_action="preserve"`:
    >>> quantize_df(df, tpq=1, zero_dur_action="preserve")
       pitch  onset  release
    0     60    0.0      0.0
    1     61    0.0      1.0
    2     62    1.0      2.0

    To remove zero-dur notes, pass `zero_dur_action="remove"` (NB the index is not
    reset):
    >>> quantize_df(df, tpq=1, zero_dur_action="remove")
       pitch  onset  release
    1     61    0.0      1.0
    2     62    1.0      2.0

    "drop" is an alias for "remove"
    >>> quantize_df(df, tpq=1, zero_dur_action="drop")
       pitch  onset  release
    1     61    0.0      1.0
    2     62    1.0      2.0
    """
    assert zero_dur_action in get_args(ZeroDurAction)

    if (
        df.attrs.get("quantized_tpq") == tpq
        and df.attrs.get("quantized_ticks_out") == ticks_out
        and df.attrs.get("quantized_zero_dur_action") == zero_dur_action
    ):
        return df

    was_sorted = df.attrs.get("sorted", False)

    onsets = np.rint(df.onset.apply(float).to_numpy() * tpq)
    releases = np.rint(df.release.apply(float).to_numpy() * tpq)
    if zero_dur_action == "min_dur":
        releases[releases == onsets] += 1
    if ticks_out:
        onsets = onsets.astype(int)
        releases = releases.astype(int)
    else:
        onsets /= tpq
        releases /= tpq
    out = pd.DataFrame(
        {
            col_name: (
                df[col_name].copy()
                if col_name not in ("onset", "release")
                else {"onset": onsets, "release": releases}[col_name]
            )
            for col_name in df.columns
        }
    )
    out.attrs = df.attrs.copy()
    out.attrs["quantized_tpq"] = tpq
    out.attrs["quantized_ticks_out"] = ticks_out
    out.attrs["quantized_zero_dur_action"] = zero_dur_action
    if zero_dur_action in {"remove", "drop"}:
        zero_dur = out["onset"] == out["release"]
        if zero_dur.any() and "type" in out.columns and "pitch" in df.columns:
            note_drops = zero_dur & (out["type"] == "note")
            if note_drops.any():
                orig = df.loc[note_drops[note_drops].index]
                out.attrs["_dropped_note_originals"] = list(
                    zip(
                        orig["onset"].astype(float),
                        orig["release"].astype(float),
                        orig["pitch"],
                    )
                )
        out = out[out["onset"] != out["release"]]
    if was_sorted:
        out = sort_df(out, force=True)
    return out
