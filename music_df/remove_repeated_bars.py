"""Remove tandem-repeated bar sequences from a music_df."""

import numpy as np
import pandas as pd

from music_df.add_feature import make_bar_explicit
from music_df.dedouble_instruments import CANDIDATE_INSTRUMENT_COLUMNS
from music_df.sort_df import sort_df
from music_df.transforms import transform

DEFAULT_MAX_BARS = 512


def _bar_fingerprints(df: pd.DataFrame) -> list[int]:
    """Return a list of fingerprint hashes, one per bar.

    Each fingerprint captures the bar's duration and all note content
    (pitch, bar-relative onset, duration, and any instrument columns),
    so two bars are "equal" iff they sound identical.
    """
    instrument_cols = [c for c in CANDIDATE_INSTRUMENT_COLUMNS if c in df.columns]

    bar_rows = df.loc[df.type == "bar"].sort_values("onset")
    bar_numbers = bar_rows["bar_number"].tolist()

    # Pre-compute bar onset/release as dicts (one pass instead of O(n) per bar)
    bar_onset = dict(zip(bar_rows["bar_number"], bar_rows["onset"]))
    bar_release = dict(zip(bar_rows["bar_number"], bar_rows["release"]))

    # Pre-extract note data as arrays for fast per-bar grouping
    notes = df[df.type == "note"]
    fp_cols = ["pitch", "onset", "release"] + instrument_cols
    note_values = {col: notes[col].values for col in fp_cols}
    note_bar_nums = notes["bar_number"].values

    fingerprints: list[int] = []
    for bar_num in bar_numbers:
        b_onset = bar_onset[bar_num]
        b_release = bar_release[bar_num]
        bar_dur = b_release - b_onset if pd.notna(b_release) else None

        mask = note_bar_nums == bar_num
        pitches = note_values["pitch"][mask]
        onsets = note_values["onset"][mask]
        releases = note_values["release"][mask]
        inst_arrays = [note_values[c][mask] for c in instrument_cols]

        note_tuples = []
        for i in range(len(pitches)):
            t = (
                pitches[i],
                round(onsets[i] - b_onset, 10),
                round(releases[i] - onsets[i], 10),
                *(arr[i] for arr in inst_arrays),
            )
            note_tuples.append(t)
        note_tuples.sort()
        fingerprints.append(hash((bar_dur, tuple(note_tuples))))

    return fingerprints


def _find_bars_to_keep(
    fingerprints: list[int],
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Return indices (into the fingerprints list) of bars to keep, plus repeat spans.

    Uses smallest-k greedy tandem-repeat removal, iterated to convergence.

    Each span is ``(pattern_start_idx, pattern_end_idx, last_copy_idx)`` —
    the first and last bar of the kept pattern, plus the last bar of the
    last repeated copy, all in terms of original bar indices.
    """
    bars = list(range(len(fingerprints)))
    fps = list(fingerprints)
    repeat_spans: list[tuple[int, int, int]] = []

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
                    repeat_spans.append(
                        (bars[i], bars[i + k - 1], bars[j - 1])
                    )
                    del fps[i + k : j]
                    del bars[i + k : j]
                    changed = True
                    found = True
                    break
            if not found:
                i += 1

    assert all(0 <= idx < len(fingerprints) for idx in bars), (
        "keep_indices out of range"
    )
    return bars, repeat_spans


def _remove_repeated_bars_diff(
    before_df: pd.DataFrame, after_df: pd.DataFrame
) -> tuple[set, set, list[tuple[float, float]]]:
    """Custom diff for remove_repeated_bars.

    The naive tuple-diff doesn't work because removing bars shifts all
    subsequent notes' timing. Instead we use the ``bars_removed`` attr
    to identify which notes were in removed bars.

    Returns ``(removed, added, diff_bounds)`` where *diff_bounds* are
    ``(start, end)`` time intervals spanning each full repeat pattern
    (original + copies), so that excerpt windows can be expanded to show
    the complete context.
    """
    bars_removed = after_df.attrs.get("bars_removed", [])
    diff_bounds = after_df.attrs.get("repeat_spans", [])
    if not bars_removed:
        return set(), set(), diff_bounds
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
    return removed_tuples, set(), diff_bounds


def _find_bars_to_keep_chunked(
    fingerprints: list[int],
    max_bars: int,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Split fingerprints into chunks of max_bars, process each independently."""
    n = len(fingerprints)
    all_keep: list[int] = []
    all_spans: list[tuple[int, int, int]] = []
    for chunk_start in range(0, n, max_bars):
        chunk_end = min(chunk_start + max_bars, n)
        chunk_fps = fingerprints[chunk_start:chunk_end]
        keep, spans = _find_bars_to_keep(chunk_fps)
        all_keep.extend(idx + chunk_start for idx in keep)
        all_spans.extend(
            (s + chunk_start, e + chunk_start, last + chunk_start)
            for s, e, last in spans
        )
    assert all(0 <= idx < n for idx in all_keep), "keep_indices out of range"
    return all_keep, all_spans


@transform(diff_func=_remove_repeated_bars_diff)
def remove_repeated_bars(
    df: pd.DataFrame, max_bars: int = DEFAULT_MAX_BARS
) -> pd.DataFrame:
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
        bar_onsets_arr = bar_rows["onset"].values
        releases = np.empty_like(bar_onsets_arr)
        releases[:-1] = bar_onsets_arr[1:]
        releases[-1] = df["release"].max()
        df.loc[bar_rows.index, "release"] = releases

    fingerprints = _bar_fingerprints(df)
    if len(fingerprints) <= max_bars:
        keep_indices, repeat_span_indices = _find_bars_to_keep(fingerprints)
    else:
        keep_indices, repeat_span_indices = _find_bars_to_keep_chunked(
            fingerprints, max_bars
        )
    bars_to_keep = [bar_numbers[i] for i in keep_indices]

    n_bars_after = len(bars_to_keep)
    n_bars_removed = n_bars_before - n_bars_after

    if n_bars_removed == 0:
        df.attrs["repeated_bars_removed"] = True
        df.attrs["n_bars_before"] = n_bars_before
        df.attrs["n_bars_after"] = n_bars_after
        df.attrs["n_bars_removed"] = 0
        df.attrs["bars_removed"] = []
        df.attrs["repeat_spans"] = []
        df.attrs["bar_onset_map"] = {}
        if "bar_number" in df.columns:
            df = df.drop(columns=["bar_number"])
        return sort_df(df, force=True)

    bars_to_keep_set = set(bars_to_keep)
    bars_removed = [b for b in bar_numbers if b not in bars_to_keep_set]

    # Keep rows that are in kept bars or precede the first bar
    default_bar_number = -1
    kept_mask = df.bar_number.isin(bars_to_keep_set) | (
        df.bar_number == default_bar_number
    )
    result = df[kept_mask].copy()

    # Compute onset shifts to close gaps from removed bars
    bar_durations = dict(zip(
        bar_rows["bar_number"],
        bar_rows["release"] - bar_rows["onset"],
    ))

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

    # Pre-compute bar onset/release dicts for O(1) lookups below
    bar_onset_dict = dict(zip(bar_rows["bar_number"], bar_rows["onset"]))
    bar_release_dict = dict(zip(bar_rows["bar_number"], bar_rows["release"]))

    # Build bar_onset_map: original bar onset -> shifted bar onset (for kept bars)
    bar_onset_map: dict[float, float] = {}
    for bar_num in bars_to_keep:
        orig_onset = bar_onset_dict[bar_num]
        bar_onset_map[orig_onset] = orig_onset - cumulative_shift[bar_num]

    # Convert index-based spans to onset/release time spans.
    # Each span is (before_start, before_end, after_start, after_end) so
    # that consumers can extract excerpts in both coordinate systems.
    # Spans from intermediate iterations whose pattern bars were later
    # removed are skipped (the outer span from the later iteration covers
    # the same region).
    repeat_spans: list[tuple[float, float, float, float]] = []
    for idx_start, idx_pattern_end, idx_span_end in repeat_span_indices:
        pat_start_bar = bar_numbers[idx_start]
        pat_end_bar = bar_numbers[idx_pattern_end]
        if pat_start_bar not in bars_to_keep_set or pat_end_bar not in bars_to_keep_set:
            continue
        before_start = bar_onset_dict[pat_start_bar]
        before_end = bar_release_dict[bar_numbers[idx_span_end]]
        after_start = before_start - cumulative_shift[pat_start_bar]
        after_end = bar_release_dict[pat_end_bar] - cumulative_shift[pat_end_bar]
        repeat_spans.append((before_start, before_end, after_start, after_end))

    result = result.drop(columns=["bar_number"])
    result = result.reset_index(drop=True)

    result.attrs = df.attrs.copy()
    result.attrs["repeated_bars_removed"] = True
    result.attrs["n_bars_before"] = n_bars_before
    result.attrs["n_bars_after"] = n_bars_after
    result.attrs["n_bars_removed"] = n_bars_removed
    result.attrs["bars_removed"] = bars_removed
    result.attrs["repeat_spans"] = repeat_spans
    result.attrs["bar_onset_map"] = bar_onset_map

    return sort_df(result, force=True)
