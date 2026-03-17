"""Remove tandem-repeated bar sequences from a music_df."""

import pandas as pd

from music_df.add_feature import make_bar_explicit
from music_df.dedouble_instruments import CANDIDATE_INSTRUMENT_COLUMNS
from music_df.sort_df import sort_df
from music_df.transforms import transform


def _bar_fingerprints(df: pd.DataFrame) -> list[int]:
    """Return a list of fingerprint hashes, one per bar.

    Each fingerprint captures the bar's duration and all note content
    (pitch, bar-relative onset, duration, and any instrument columns),
    so two bars are "equal" iff they sound identical.
    """
    instrument_cols = [c for c in CANDIDATE_INSTRUMENT_COLUMNS if c in df.columns]

    bar_mask = df.type == "bar"
    bar_rows = df.loc[bar_mask].sort_values("onset")
    bar_numbers = bar_rows["bar_number"].tolist()

    fingerprints: list[int] = []
    for bar_num in bar_numbers:
        bar_onset = bar_rows.loc[bar_rows.bar_number == bar_num, "onset"].iloc[0]
        bar_release = bar_rows.loc[bar_rows.bar_number == bar_num, "release"].iloc[0]
        bar_dur = bar_release - bar_onset if pd.notna(bar_release) else None

        notes = df[(df.bar_number == bar_num) & (df.type == "note")]
        note_tuples = []
        for _, row in notes.iterrows():
            t = (
                row.pitch,
                round(row.onset - bar_onset, 10),
                round(row.release - row.onset, 10),
                *(row[c] for c in instrument_cols),
            )
            note_tuples.append(t)
        note_tuples.sort()
        fingerprints.append(hash((bar_dur, tuple(note_tuples))))

    return fingerprints


def _find_bars_to_keep(fingerprints: list[int]) -> list[int]:
    """Return indices (into the fingerprints list) of bars to keep.

    Uses smallest-k greedy tandem-repeat removal, iterated to convergence.
    """
    bars = list(range(len(fingerprints)))
    fps = list(fingerprints)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(fps):
            found = False
            for k in range(1, (len(fps) - i) // 2 + 1):
                if fps[i : i + k] == fps[i + k : i + 2 * k]:
                    j = i + 2 * k
                    while j + k <= len(fps) and fps[i : i + k] == fps[j : j + k]:
                        j += k
                    del fps[i + k : j]
                    del bars[i + k : j]
                    changed = True
                    found = True
                    break
            if not found:
                i += 1

    return bars


def _remove_repeated_bars_diff(
    before_df: pd.DataFrame, after_df: pd.DataFrame
) -> tuple[set, set]:
    """Custom diff for remove_repeated_bars.

    The naive tuple-diff doesn't work because removing bars shifts all
    subsequent notes' timing. Instead we use the ``bars_removed`` attr
    to identify which notes were in removed bars.
    """
    bars_removed = after_df.attrs.get("bars_removed", [])
    if not bars_removed:
        return set(), set()
    before_with_bars = make_bar_explicit(before_df)
    removed_notes = before_with_bars[
        (before_with_bars["type"] == "note")
        & before_with_bars["bar_number"].isin(set(bars_removed))
    ]
    removed_tuples = set(
        removed_notes[["onset", "release", "pitch"]].itertuples(
            index=False, name=None
        )
    )
    return removed_tuples, set()


@transform(diff_func=_remove_repeated_bars_diff)
def remove_repeated_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Remove tandem-repeated bar sequences from a music_df.

    A "tandem repeat" is a consecutive repetition of a bar sequence.
    For example, if bars ABAB appear, the second AB is removed.
    Uses smallest-k greedy strategy with iteration until convergence.

    Requires barlines (rows with ``type == "bar"``); raises if none exist.

    We store the following attributes in ``.attrs``:

    - ``repeated_bars_removed`` (bool)
    - ``n_bars_before`` (int)
    - ``n_bars_after`` (int)
    - ``n_bars_removed`` (int)

    >>> import numpy as np, pandas as pd
    >>> pd.set_option("display.width", 200)
    >>> pd.set_option("display.max_columns", None)

    **k=1 repeat (AA -> A):**

    >>> df_aa = pd.DataFrame({
    ...     "type":    ["bar",  "note", "bar",  "note", "bar"],
    ...     "onset":   [0.0,    0.0,    4.0,    4.0,    8.0],
    ...     "release": [4.0,    1.0,    8.0,    5.0,    12.0],
    ...     "pitch":   [np.nan, 60.0,   np.nan, 60.0,   np.nan],
    ... })
    >>> result = remove_repeated_bars(df_aa)
    >>> result[["type", "onset", "release", "pitch"]]  # doctest: +NORMALIZE_WHITESPACE
      type  onset  release  pitch
    0  bar    0.0      4.0    NaN
    1 note    0.0      1.0   60.0
    2  bar    4.0      8.0    NaN
    >>> result.attrs["n_bars_removed"]
    1

    **k=2 repeat (ABAB -> AB):**

    >>> df_abab = pd.DataFrame({
    ...     "type":    ["bar",  "note", "bar",  "note", "bar",  "note", "bar",  "note", "bar"],
    ...     "onset":   [0.0,    0.0,    4.0,    4.0,    8.0,    8.0,    12.0,   12.0,   16.0],
    ...     "release": [4.0,    1.0,    8.0,    5.0,    12.0,   9.0,    16.0,   13.0,   20.0],
    ...     "pitch":   [np.nan, 60.0,   np.nan, 62.0,   np.nan, 60.0,   np.nan, 62.0,   np.nan],
    ... })
    >>> result = remove_repeated_bars(df_abab)
    >>> result[["type", "onset", "release", "pitch"]]  # doctest: +NORMALIZE_WHITESPACE
      type  onset  release  pitch
    0  bar    0.0      4.0    NaN
    1 note    0.0      1.0   60.0
    2  bar    4.0      8.0    NaN
    3 note    4.0      5.0   62.0
    4  bar    8.0     12.0    NaN
    >>> result.attrs["n_bars_removed"]
    2

    **Multiple repetitions (ABABAB -> AB):**

    >>> df_ababab = pd.DataFrame({
    ...     "type":    ["bar",  "note", "bar",  "note",
    ...                 "bar",  "note", "bar",  "note",
    ...                 "bar",  "note", "bar",  "note", "bar"],
    ...     "onset":   [0.0,    0.0,    4.0,    4.0,
    ...                 8.0,    8.0,    12.0,   12.0,
    ...                 16.0,   16.0,   20.0,   20.0,   24.0],
    ...     "release": [4.0,    1.0,    8.0,    5.0,
    ...                 12.0,   9.0,    16.0,   13.0,
    ...                 20.0,   17.0,   24.0,   21.0,   28.0],
    ...     "pitch":   [np.nan, 60.0,   np.nan, 62.0,
    ...                 np.nan, 60.0,   np.nan, 62.0,
    ...                 np.nan, 60.0,   np.nan, 62.0,   np.nan],
    ... })
    >>> result = remove_repeated_bars(df_ababab)
    >>> result.attrs["n_bars_removed"]
    4

    **Nested repeats requiring iteration (AABB -> AB):**

    >>> df_aabb = pd.DataFrame({
    ...     "type":    ["bar",  "note", "bar",  "note",
    ...                 "bar",  "note", "bar",  "note", "bar"],
    ...     "onset":   [0.0,    0.0,    4.0,    4.0,
    ...                 8.0,    8.0,    12.0,   12.0,   16.0],
    ...     "release": [4.0,    1.0,    8.0,    5.0,
    ...                 12.0,   9.0,    16.0,   13.0,   20.0],
    ...     "pitch":   [np.nan, 60.0,   np.nan, 60.0,
    ...                 np.nan, 62.0,   np.nan, 62.0,   np.nan],
    ... })
    >>> result = remove_repeated_bars(df_aabb)
    >>> result[["type", "onset", "release", "pitch"]]  # doctest: +NORMALIZE_WHITESPACE
      type  onset  release  pitch
    0  bar    0.0      4.0    NaN
    1 note    0.0      1.0   60.0
    2  bar    4.0      8.0    NaN
    3 note    4.0      5.0   62.0
    4  bar    8.0     12.0    NaN
    >>> result.attrs["n_bars_removed"]
    2

    **No repeats (ABC -> ABC, unchanged):**

    >>> df_abc = pd.DataFrame({
    ...     "type":    ["bar",  "note", "bar",  "note", "bar",  "note", "bar"],
    ...     "onset":   [0.0,    0.0,    4.0,    4.0,    8.0,    8.0,    12.0],
    ...     "release": [4.0,    1.0,    8.0,    5.0,    12.0,   10.0,   16.0],
    ...     "pitch":   [np.nan, 60.0,   np.nan, 62.0,   np.nan, 64.0,   np.nan],
    ... })
    >>> result = remove_repeated_bars(df_abc)
    >>> result.attrs["n_bars_removed"]
    0

    **Multi-track bars:**

    >>> df_multi = pd.DataFrame({
    ...     "type":    ["bar",  "note", "note", "bar",  "note", "note", "bar"],
    ...     "onset":   [0.0,    0.0,    0.0,    4.0,    4.0,    4.0,    8.0],
    ...     "release": [4.0,    1.0,    2.0,    8.0,    5.0,    6.0,    12.0],
    ...     "pitch":   [np.nan, 60.0,   64.0,   np.nan, 60.0,   64.0,   np.nan],
    ...     "track":   [np.nan, 0.0,    1.0,    np.nan, 0.0,    1.0,    np.nan],
    ... })
    >>> result = remove_repeated_bars(df_multi)
    >>> result.attrs["n_bars_removed"]
    1
    """
    bar_mask = df.type == "bar"
    if not bar_mask.any():
        raise ValueError("No bars found")

    df = df.copy()
    df.attrs = df.attrs.copy()

    if "bar_number" not in df.columns:
        df = make_bar_explicit(df)

    bar_rows = df.loc[df.type == "bar"].sort_values("onset")
    bar_numbers = bar_rows["bar_number"].tolist()
    n_bars_before = len(bar_numbers)

    # Compute bar releases if missing
    if bar_rows["release"].isna().any():
        bar_onsets = bar_rows["onset"].tolist()
        max_release = df["release"].max()
        for idx, bar_num in enumerate(bar_numbers):
            bar_release = (
                bar_onsets[idx + 1] if idx + 1 < len(bar_onsets) else max_release
            )
            df.loc[
                (df.bar_number == bar_num) & (df.type == "bar"), "release"
            ] = bar_release

    fingerprints = _bar_fingerprints(df)
    keep_indices = _find_bars_to_keep(fingerprints)
    bars_to_keep = [bar_numbers[i] for i in keep_indices]

    n_bars_after = len(bars_to_keep)
    n_bars_removed = n_bars_before - n_bars_after

    if n_bars_removed == 0:
        df.attrs["repeated_bars_removed"] = True
        df.attrs["n_bars_before"] = n_bars_before
        df.attrs["n_bars_after"] = n_bars_after
        df.attrs["n_bars_removed"] = 0
        df.attrs["bars_removed"] = []
        if "bar_number" in df.columns:
            df = df.drop(columns=["bar_number"])
        return sort_df(df)

    bars_to_keep_set = set(bars_to_keep)
    bars_removed = [b for b in bar_numbers if b not in bars_to_keep_set]

    # Keep rows that are in kept bars or precede the first bar
    default_bar_number = -1
    kept_mask = df.bar_number.isin(bars_to_keep_set) | (
        df.bar_number == default_bar_number
    )
    result = df[kept_mask].copy()

    # Compute onset shifts to close gaps from removed bars
    bar_durations = {}
    for bar_num in bar_numbers:
        bar_row = df[(df.bar_number == bar_num) & (df.type == "bar")]
        if not bar_row.empty:
            row = bar_row.iloc[0]
            bar_durations[bar_num] = row.release - row.onset

    cumulative_shift: dict[int, float] = {}
    shift = 0.0
    for bar_num in bar_numbers:
        if bar_num not in bars_to_keep_set:
            shift += bar_durations.get(bar_num, 0.0)
        else:
            cumulative_shift[bar_num] = shift
    cumulative_shift[default_bar_number] = 0.0

    result["onset"] = result["onset"] - result["bar_number"].map(cumulative_shift)
    release_mask = result["release"].notna()
    result.loc[release_mask, "release"] = (
        result.loc[release_mask, "release"]
        - result.loc[release_mask, "bar_number"].map(cumulative_shift)
    )

    result = result.drop(columns=["bar_number"])
    result = result.reset_index(drop=True)

    result.attrs = df.attrs.copy()
    result.attrs["repeated_bars_removed"] = True
    result.attrs["n_bars_before"] = n_bars_before
    result.attrs["n_bars_after"] = n_bars_after
    result.attrs["n_bars_removed"] = n_bars_removed
    result.attrs["bars_removed"] = bars_removed

    return sort_df(result)
