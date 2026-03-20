"""
Instrument-aware dedoubling via suffix array + LCP.

Detects cross-instrument doublings — runs of >= n consecutive matching notes
played by different instruments — and removes one copy. Supports both exact
pitch matching (``dedouble_unisons_across_instruments``) and octave-equivalent matching
(``dedouble_octaves``).

Also provides within-instrument octave dedoubling
(``dedouble_octaves_within_instrument``), which uses a streak-tracking
algorithm over onset-grouped chords to find notes doubled at the octave
within a single instrument (e.g., piano playing C4+C5 simultaneously).

Requires the optional ``pydivsufsort`` dependency for cross-instrument
functions (install with ``pip install music_df[doublings]``).
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field

from music_df.transforms import transform
from typing import Callable, Sequence

import numpy as np
import pandas as pd

CANDIDATE_INSTRUMENT_COLUMNS = ("instrument", "part", "track", "channel")
_INF = float("inf")


def _min_sounding_pitch_at_onsets(
    onset_qs: np.ndarray, release_qs: np.ndarray, pitches: np.ndarray
) -> dict[float, float]:
    """For each unique quantized onset, find the minimum pitch sounding at that time.

    A note is "sounding" at time *t* when onset_q <= t and release_q > t.
    Uses an O(N log N) sweep-line with a min-heap and lazy deletion.
    """
    unique_onsets = np.unique(onset_qs)
    order = np.argsort(onset_qs)
    heap: list[tuple[float, float]] = []  # (pitch, release_q)
    note_idx = 0
    result: dict[float, float] = {}
    for t in unique_onsets:
        while note_idx < len(order) and onset_qs[order[note_idx]] <= t:
            heapq.heappush(
                heap, (pitches[order[note_idx]], release_qs[order[note_idx]])
            )
            note_idx += 1
        while heap and heap[0][1] <= t:
            heapq.heappop(heap)
        if heap:
            result[t] = heap[0][0]
    return result


def _resolve_instrument_columns(
    df: pd.DataFrame, instrument_columns: Sequence[str] | None
) -> list[str]:
    """Auto-detect or validate instrument columns."""
    if instrument_columns is None:
        instrument_columns = [
            c for c in CANDIDATE_INSTRUMENT_COLUMNS if c in df.columns
        ]
    if not instrument_columns:
        raise ValueError(
            "No instrument columns found. Pass instrument_columns explicitly "
            f"or ensure df has at least one of {CANDIDATE_INSTRUMENT_COLUMNS}."
        )
    return list(instrument_columns)


def _prepare_notes(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None,
    quantize: bool,
    ticks_per_quarter: int,
    release_ticks_per_quarter: int | None,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Separate notes/non-notes, resolve instrument columns, quantize.

    Adds ``_inst_key``, ``_onset_q``, ``_release_q`` columns to the note df.

    Returns
    -------
    note_df : pd.DataFrame
        Notes with helper columns added.
    n_non_notes : int
        Count of non-note rows in the original df.
    instrument_columns : list[str]
        Resolved instrument columns.
    """
    instrument_columns = _resolve_instrument_columns(df, instrument_columns)

    note_mask = df["type"] == "note"
    note_df = df[note_mask].copy()
    n_non_notes = int((~note_mask).sum())

    if note_df.empty:
        return note_df, n_non_notes, instrument_columns

    release_tpq = (
        release_ticks_per_quarter
        if release_ticks_per_quarter is not None
        else ticks_per_quarter
    )
    if quantize:
        onset_vals = (note_df["onset"] * ticks_per_quarter).round()
        release_vals = (note_df["release"] * release_tpq).round()
    else:
        onset_vals = note_df["onset"]
        release_vals = note_df["release"]

    if len(instrument_columns) == 1:
        inst_key = note_df[instrument_columns[0]].astype(str)
    else:
        inst_key = (
            note_df[instrument_columns].astype(str).agg("|".join, axis=1)
        )
    note_df["_inst_key"] = inst_key
    note_df["_onset_q"] = onset_vals
    note_df["_release_q"] = release_vals

    return note_df, n_non_notes, instrument_columns


def _find_doublings_drop_indices(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None,
    min_length: int,
    quantize: bool,
    ticks_per_quarter: int,
    pitch_key_mode: str,
    drop_selector: Callable[[int, int, float, float], int],
    release_ticks_per_quarter: int | None = None,
    use_bass_detection: bool = False,
) -> tuple[set[int], pd.DataFrame, pd.DataFrame, int, int, list[str], dict[float, float] | None]:
    """Core suffix-array doubling detection, returning raw drop indices.

    Returns
    -------
    drop_indices : set[int]
        DataFrame indices to drop.
    note_df : pd.DataFrame
        Notes with ``_inst_key``, ``_onset_q``, ``_release_q`` columns.
    df : pd.DataFrame
        Copy of input df.
    n_undedoubled_notes : int
    n_non_notes : int
    instrument_columns : list[str]
    min_sounding : dict[float, float] | None
        When *use_bass_detection* is True, the min sounding pitch at each
        onset; None otherwise.
    """
    try:
        from pydivsufsort import divsufsort, kasai
    except ImportError as exc:
        raise ImportError(
            "pydivsufsort is required for dedouble_unisons_across_instruments. "
            "Install it with: pip install music_df[doublings]"
        ) from exc

    df = df.copy()
    n_undedoubled_notes = int((df.type == "note").sum())

    note_df, n_non_notes, instrument_columns = _prepare_notes(
        df, instrument_columns, quantize, ticks_per_quarter,
        release_ticks_per_quarter,
    )

    if note_df.empty:
        return set(), note_df, df, n_undedoubled_notes, n_non_notes, instrument_columns, None

    # --- Compute min sounding pitch if needed ---
    min_sounding: dict[float, float] | None = None
    if use_bass_detection:
        min_sounding = _min_sounding_pitch_at_onsets(
            note_df["_onset_q"].values,
            note_df["_release_q"].values,
            note_df["pitch"].values,
        )

    # --- Compute pitch keys vectorized ---
    pitch_vals = note_df["pitch"].values
    if pitch_key_mode == "mod12":
        pitch_keys = pitch_vals % 12
    else:
        pitch_keys = pitch_vals.copy()
    note_df["_pitch_key"] = pitch_keys

    # --- Assign token IDs via np.unique on structured array ---
    onset_q = note_df["_onset_q"].values
    release_q = note_df["_release_q"].values
    struct = np.empty(len(note_df), dtype=[
        ("onset_q", np.float64), ("release_q", np.float64),
        ("pitch_key", np.float64),
    ])
    struct["onset_q"] = onset_q
    struct["release_q"] = release_q
    struct["pitch_key"] = pitch_keys
    _, token_ids = np.unique(struct, return_inverse=True)
    note_df["_token"] = token_ids

    # --- Precompute per-note min_sounding values for suffix array positions ---
    if use_bass_detection:
        onset_q_col_raw = note_df["_onset_q"].values
        unique_oqs, inverse = np.unique(onset_q_col_raw, return_inverse=True)
        ms_for_unique = np.array(
            [min_sounding.get(oq, _INF) for oq in unique_oqs]
        )
        min_sounding_col = ms_for_unique[inverse]

    # --- Sort once, then groups are pre-sorted ---
    note_df = note_df.sort_values(["_inst_key", "onset", "pitch"])
    orig_indices = note_df.index.values

    inst_key_vals = note_df["_inst_key"].values
    token_vals = note_df["_token"].values
    pitch_col = note_df["pitch"].values
    onset_q_col = note_df["_onset_q"].values

    # --- Build per-group arrays without iterrows ---
    unique_insts, group_starts = np.unique(inst_key_vals, return_index=True)
    group_ends = np.append(group_starts[1:], len(inst_key_vals))

    sequences: list[np.ndarray] = []
    index_maps: list[np.ndarray] = []
    inst_labels: list[np.ndarray] = []
    pitch_arrays: list[np.ndarray] = []
    min_sounding_arrays: list[np.ndarray] = [] if use_bass_detection else None
    sentinel = -1

    # Need to recompute min_sounding_col in sort order if bass detection
    if use_bass_detection:
        # min_sounding_col was computed in pre-sort order; reindex to sorted
        sort_order = note_df["_onset_q"].values  # already sorted note_df
        unique_oqs_s, inverse_s = np.unique(sort_order, return_inverse=True)
        ms_for_unique_s = np.array(
            [min_sounding.get(oq, _INF) for oq in unique_oqs_s]
        )
        min_sounding_sorted = ms_for_unique_s[inverse_s]

    for inst_idx in range(len(unique_insts)):
        start, end = group_starts[inst_idx], group_ends[inst_idx]
        grp_tokens = token_vals[start:end]
        grp_indices = orig_indices[start:end]
        grp_pitches = pitch_col[start:end]

        n = end - start
        sequences.append(grp_tokens.astype(np.int64))
        sequences.append(np.array([sentinel], dtype=np.int64))
        sentinel -= 1

        index_maps.append(grp_indices)
        index_maps.append(np.array([-1]))

        inst_labels.append(np.full(n, inst_idx, dtype=np.int64))
        inst_labels.append(np.array([-1], dtype=np.int64))

        pitch_arrays.append(grp_pitches.astype(np.float64))
        pitch_arrays.append(np.array([np.nan]))

        if use_bass_detection:
            min_sounding_arrays.append(
                min_sounding_sorted[start:end].astype(np.float64)
            )
            min_sounding_arrays.append(np.array([_INF]))

    # --- Concatenate ---
    concatenated = np.concatenate(sequences)
    all_indices = np.concatenate(index_maps)
    all_inst = np.concatenate(inst_labels)
    all_pitches = np.concatenate(pitch_arrays)
    all_min_sounding = (
        np.concatenate(min_sounding_arrays) if use_bass_detection else None
    )

    # --- Suffix array + LCP ---
    sa = divsufsort(concatenated)
    lcp = kasai(concatenated, sa)

    # --- Scan for cross-instrument doublings (vectorized filter) ---
    lcp_arr = np.asarray(lcp)
    sa_arr = np.asarray(sa)
    n_sa = len(sa_arr)

    # Filter to candidates where lcp >= min_length
    cand_mask = lcp_arr[: n_sa - 1] >= min_length
    cand_idx = np.where(cand_mask)[0]

    if len(cand_idx) == 0:
        return set(), note_df, df, n_undedoubled_notes, n_non_notes, instrument_columns, min_sounding

    pos_a_arr = sa_arr[cand_idx]
    pos_b_arr = sa_arr[cand_idx + 1]
    inst_a_arr = all_inst[pos_a_arr]
    inst_b_arr = all_inst[pos_b_arr]

    # Keep only cross-instrument, non-sentinel pairs
    valid = (inst_a_arr >= 0) & (inst_b_arr >= 0) & (inst_a_arr != inst_b_arr)
    cand_idx = cand_idx[valid]
    pos_a_arr = pos_a_arr[valid]
    pos_b_arr = pos_b_arr[valid]
    inst_a_arr = inst_a_arr[valid]
    inst_b_arr = inst_b_arr[valid]

    drop_indices: set[int] = set()
    total_len = len(all_inst)

    for k in range(len(cand_idx)):
        match_len = int(lcp_arr[cand_idx[k]])
        pos_a = int(pos_a_arr[k])
        pos_b = int(pos_b_arr[k])
        inst_a = int(inst_a_arr[k])
        inst_b = int(inst_b_arr[k])

        passage_mean_a = all_pitches[pos_a:pos_a + match_len].mean()
        passage_mean_b = all_pitches[pos_b:pos_b + match_len].mean()

        if use_bass_detection:
            is_bass = False
            for offset in range(match_len):
                lower_pitch = (
                    all_pitches[pos_a + offset]
                    if all_pitches[pos_a + offset] < all_pitches[pos_b + offset]
                    else all_pitches[pos_b + offset]
                )
                if lower_pitch <= all_min_sounding[pos_a + offset]:
                    is_bass = True
                    break
            if is_bass:
                drop_inst = (
                    inst_a if passage_mean_a > passage_mean_b else inst_b
                )
            else:
                drop_inst = (
                    inst_a if passage_mean_a < passage_mean_b else inst_b
                )
        else:
            drop_inst = drop_selector(
                inst_a, inst_b, passage_mean_a, passage_mean_b
            )
        drop_pos = pos_a if drop_inst == inst_a else pos_b

        # Vectorized offset collection
        end = min(drop_pos + match_len, total_len)
        offsets = np.arange(drop_pos, end)
        valid_mask = all_inst[offsets] >= 0
        orig = all_indices[offsets[valid_mask]]
        valid_orig = orig[orig >= 0]
        drop_indices.update(valid_orig.tolist())

    return drop_indices, note_df, df, n_undedoubled_notes, n_non_notes, instrument_columns, min_sounding


def _find_doublings(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None,
    min_length: int,
    quantize: bool,
    ticks_per_quarter: int,
    pitch_key_mode: str,
    drop_selector: Callable[[int, int, float, float], int],
    release_ticks_per_quarter: int | None = None,
) -> pd.DataFrame:
    """Shared core for exact and octave dedoubling.

    Parameters
    ----------
    pitch_key_mode : str
        ``"identity"`` for exact pitch matching, ``"mod12"`` for
        octave-equivalent matching.
    drop_selector : callable
        ``(inst_a, inst_b, passage_mean_a, passage_mean_b) -> inst_to_drop``
    """
    drop_indices, note_df, df, n_undedoubled_notes, n_non_notes, inst_cols, _ = (
        _find_doublings_drop_indices(
            df, instrument_columns, min_length, quantize, ticks_per_quarter,
            pitch_key_mode, drop_selector, release_ticks_per_quarter,
        )
    )
    return _build_output(df, drop_indices, n_undedoubled_notes, n_non_notes)


def _dedouble_diff(before_df: pd.DataFrame, after_df: pd.DataFrame) -> tuple[set, set]:
    """Multiset diff for dedouble transforms.

    The default set-based diff misses removals when another note with the
    same (onset, release, pitch) remains (e.g., an octave pair in track 6
    is removed but track 8 still has the same pitch at that onset).
    """
    _KEY_COLS = ("onset", "release", "pitch")

    def _note_counter(df: pd.DataFrame) -> Counter:
        notes = df[df["type"] == "note"]
        cols = [c for c in _KEY_COLS if c in notes.columns]
        return Counter(notes[cols].itertuples(index=False, name=None))

    before_counts = _note_counter(before_df)
    after_counts = _note_counter(after_df)
    removed = set(before_counts - after_counts)
    added = set(after_counts - before_counts)
    return removed, added


@transform(diff_func=_dedouble_diff)
def dedouble_unisons_across_instruments(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None = None,
    min_length: int = 2,
    quantize: bool = True,
    ticks_per_quarter: int = 16,
    release_ticks_per_quarter: int | None = None,
) -> pd.DataFrame:
    """Remove cross-instrument doublings from a music_df.

    Two instruments "double" each other when they play a run of >= *min_length*
    consecutive identical (onset, release, pitch) triples.  This function finds
    all such runs using a suffix-array / LCP approach, then drops the
    higher-sorted instrument's copy of each doubled passage.

    Parameters
    ----------
    df : pd.DataFrame
        A music_df with at least ``type``, ``onset``, ``release``, ``pitch``
        columns and one or more instrument-identifying columns.
    instrument_columns : sequence of str or None
        Columns that jointly identify an instrument.  If *None*, auto-detected
        from ``CANDIDATE_INSTRUMENT_COLUMNS``.
    min_length : int
        Minimum consecutive matching notes to count as a doubling.
    quantize : bool
        Round onset/release to a grid before comparison.
    ticks_per_quarter : int
        Grid resolution for onsets when *quantize* is True.
    release_ticks_per_quarter : int or None
        Grid resolution for releases when *quantize* is True. If *None*,
        uses *ticks_per_quarter*. A lower value gives a coarser grid,
        useful when releases are less precise than onsets.

    Returns
    -------
    pd.DataFrame
        The dedoubled dataframe, with ``original_index`` column and attrs
        ``n_undedoubled_notes``, ``n_dedoubled_notes``,
        ``dedoubled_instruments``.

    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     "type":    ["bar"] + ["note"]*6,
    ...     "track":   [np.nan, 1, 1, 1, 2, 2, 2],
    ...     "pitch":   [np.nan, 60, 62, 64, 60, 62, 64],
    ...     "onset":   [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    ...     "release": [np.nan, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    ... })
    >>> result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
    >>> result.attrs["n_undedoubled_notes"]
    6
    >>> result.attrs["n_dedoubled_notes"]
    3
    >>> sorted(result[result.type == "note"]["track"].unique())
    [1.0]
    """
    return _find_doublings(
        df, instrument_columns, min_length, quantize, ticks_per_quarter,
        pitch_key_mode="identity",
        drop_selector=lambda a, b, _ma, _mb: b if a < b else a,
        release_ticks_per_quarter=release_ticks_per_quarter,
    )


def _find_polyphonic_octave_doublings(
    note_df: pd.DataFrame,
    polyphonic_insts: set[str],
    already_dropped: set[int],
    min_length: int,
    min_sounding_pitch: dict[float, float],
    idx_to_onset_q: dict[int, float],
    idx_to_pitch: dict[int, float],
    match_releases: bool = True,
    max_gap_onsets: int = 0,
    octave_intervals: set[int] = frozenset({12, 24, 36}),
) -> set[int]:
    """Find octave doublings involving polyphonic instruments via onset-grouped
    cross-instrument matching.

    For each instrument pair where at least one is polyphonic, groups notes by
    onset, finds cross-instrument octave pairs, and tracks streaks using the
    same ``_Streak`` / ``_finalize_streak`` logic as within-instrument
    dedoubling.
    """
    if already_dropped:
        mask = ~note_df.index.isin(already_dropped)
        active = note_df[mask]
    else:
        active = note_df
    if active.empty:
        return set()

    drop_indices: set[int] = set()
    unique_insts = active["_inst_key"].unique()

    # Pre-extract sorted arrays per instrument (once), keyed by inst_key
    inst_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for inst_key in unique_insts:
        inst_notes = active[active["_inst_key"] == inst_key].sort_values(
            ["_onset_q", "pitch"]
        )
        inst_arrays[inst_key] = (
            inst_notes["pitch"].values,
            inst_notes["_onset_q"].values,
            inst_notes["_release_q"].values,
            inst_notes.index.values,
        )

    for i, inst_a in enumerate(unique_insts):
        for inst_b in unique_insts[i + 1:]:
            if inst_a not in polyphonic_insts and inst_b not in polyphonic_insts:
                continue

            pitches_a, onsets_a, releases_a, indices_a = inst_arrays[inst_a]
            pitches_b, onsets_b, releases_b, indices_b = inst_arrays[inst_b]

            shared_onsets = np.intersect1d(
                np.unique(onsets_a), np.unique(onsets_b)
            )
            if len(shared_onsets) < min_length:
                continue

            shared_onsets.sort()

            # Precompute onset boundaries via searchsorted
            a_starts = np.searchsorted(onsets_a, shared_onsets, side="left")
            a_ends = np.searchsorted(onsets_a, shared_onsets, side="right")
            b_starts = np.searchsorted(onsets_b, shared_onsets, side="left")
            b_ends = np.searchsorted(onsets_b, shared_onsets, side="right")

            active_streaks: list[_Streak] = []

            for o_idx in range(len(shared_onsets)):
                sa, ea = a_starts[o_idx], a_ends[o_idx]
                sb, eb = b_starts[o_idx], b_ends[o_idx]

                # Find all cross-instrument octave pairs at this onset
                pairs_at_onset: list[tuple[int, int, int, float, float]] = []
                for ia in range(sa, ea):
                    for ib in range(sb, eb):
                        interval = int(abs(pitches_a[ia] - pitches_b[ib]))
                        if interval not in octave_intervals:
                            continue
                        if match_releases and releases_a[ia] != releases_b[ib]:
                            continue
                        if pitches_a[ia] < pitches_b[ib]:
                            lo_idx = int(indices_a[ia])
                            hi_idx = int(indices_b[ib])
                            lo_p = float(pitches_a[ia])
                            hi_p = float(pitches_b[ib])
                        else:
                            lo_idx = int(indices_b[ib])
                            hi_idx = int(indices_a[ia])
                            lo_p = float(pitches_b[ib])
                            hi_p = float(pitches_a[ia])
                        pairs_at_onset.append((
                            interval, lo_idx, hi_idx, lo_p, hi_p,
                        ))

                # Match active streaks to pairs (same logic as within-instrument)
                consumed: set[int] = set()
                streaks_to_remove: list[int] = []

                for s_idx, streak in enumerate(active_streaks):
                    best_pair_idx = None
                    best_dist = float("inf")
                    for p_idx, (interval, _li, _ui, lp, _up) in enumerate(
                        pairs_at_onset
                    ):
                        if p_idx in consumed:
                            continue
                        if interval != streak.interval:
                            continue
                        dist = abs(lp - streak.last_lower_pitch)
                        if dist < best_dist:
                            best_dist = dist
                            best_pair_idx = p_idx

                    if best_pair_idx is not None:
                        consumed.add(best_pair_idx)
                        _, li, ui, lp, up = pairs_at_onset[best_pair_idx]
                        streak.lower_indices.append(li)
                        streak.upper_indices.append(ui)
                        streak.last_lower_pitch = lp
                        streak.length += 1
                        streak.gap = 0
                    else:
                        streak.gap += 1
                        if streak.gap > max_gap_onsets:
                            if streak.length >= min_length:
                                _finalize_streak(
                                    streak, min_sounding_pitch,
                                    idx_to_onset_q, idx_to_pitch,
                                    drop_indices,
                                )
                            streaks_to_remove.append(s_idx)

                for s_idx in reversed(streaks_to_remove):
                    del active_streaks[s_idx]

                for p_idx, (interval, li, ui, lp, up) in enumerate(
                    pairs_at_onset
                ):
                    if p_idx in consumed:
                        continue
                    active_streaks.append(_Streak(
                        interval=interval,
                        lower_indices=[li],
                        upper_indices=[ui],
                        last_lower_pitch=lp,
                        length=1,
                        gap=0,
                    ))

            for streak in active_streaks:
                if streak.length >= min_length:
                    _finalize_streak(
                        streak, min_sounding_pitch, idx_to_onset_q,
                        idx_to_pitch, drop_indices,
                    )

    return drop_indices


@transform(diff_func=_dedouble_diff)
def dedouble_octaves(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None = None,
    min_length: int = 3,
    quantize: bool = True,
    ticks_per_quarter: int = 16,
    release_ticks_per_quarter: int | None = None,
) -> pd.DataFrame:
    """Remove cross-instrument octave doublings from a music_df.

    Like ``dedouble_unisons_across_instruments`` but matches by pitch class
    (pitch % 12) instead of exact pitch.  Defaults to *min_length=3* to
    reduce false positives from contrary motion.

    Uses bass-detection to decide which voice to keep: if any note in the
    doubled passage is the lowest sounding note at its onset, the passage
    is treated as a bass doubling (keep lower, drop higher); otherwise it
    is treated as a melody doubling (keep higher, drop lower).

    Uses a two-pass approach: first a suffix-array pass (fast, handles
    monophonic cases), then an onset-grouped cross-instrument pass for
    pairs involving polyphonic instruments where the suffix array misses
    doublings due to extra tokens from chords.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "type":    ["note"]*6,
    ...     "track":   [1, 1, 1, 2, 2, 2],
    ...     "pitch":   [60, 62, 64, 72, 74, 76],
    ...     "onset":   [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    ...     "release": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    ... })
    >>> result = dedouble_octaves(df, instrument_columns=["track"])
    >>> result.attrs["n_dedoubled_notes"]
    3
    """
    # Pass 1: suffix array (handles monophonic instruments)
    drop_indices, note_df, df_copy, n_undedoubled, n_non_notes, inst_cols, min_sounding = (
        _find_doublings_drop_indices(
            df, instrument_columns, min_length, quantize, ticks_per_quarter,
            pitch_key_mode="mod12",
            drop_selector=lambda a, b, ma, mb: b if a < b else a,
            release_ticks_per_quarter=release_ticks_per_quarter,
            use_bass_detection=True,
        )
    )

    # Build idx_to_* maps from the returned note_df (cheap O(N))
    if not note_df.empty:
        idx_to_onset_q = dict(zip(note_df.index, note_df["_onset_q"]))
        idx_to_pitch = dict(zip(note_df.index, note_df["pitch"]))
    else:
        min_sounding = {}
        idx_to_onset_q = {}
        idx_to_pitch = {}

    # Pass 2: polyphonic fallback
    if not note_df.empty:
        # Fast O(N) check: any instrument with >1 note at same onset?
        sorted_df = note_df.sort_values(["_inst_key", "_onset_q"])
        ik = sorted_df["_inst_key"].values
        oq = sorted_df["_onset_q"].values
        is_dup = (ik[:-1] == ik[1:]) & (oq[:-1] == oq[1:])
        has_poly = np.any(is_dup)
        if has_poly:
            dup_indices = np.where(is_dup)[0]
            polyphonic_insts = set(ik[dup_indices])
        else:
            polyphonic_insts = set()
        if polyphonic_insts:
            poly_drops = _find_polyphonic_octave_doublings(
                note_df,
                polyphonic_insts,
                already_dropped=drop_indices,
                min_length=min_length,
                min_sounding_pitch=min_sounding,
                idx_to_onset_q=idx_to_onset_q,
                idx_to_pitch=idx_to_pitch,
            )
            drop_indices |= poly_drops

    return _build_output(df_copy, drop_indices, n_undedoubled, n_non_notes)


# ---------------------------------------------------------------------------
# Within-instrument octave dedoubling
# ---------------------------------------------------------------------------

@dataclass
class _Streak:
    """Tracks an active run of octave-doubled onsets within one instrument."""

    interval: int = 0
    lower_indices: list[int] = field(default_factory=list)
    upper_indices: list[int] = field(default_factory=list)
    last_lower_pitch: float = 0.0
    length: int = 0
    gap: int = 0


@transform(diff_func=_dedouble_diff)
def dedouble_octaves_within_instrument(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None = None,
    min_length: int = 3,
    quantize: bool = True,
    ticks_per_quarter: int = 16,
    release_ticks_per_quarter: int | None = None,
    match_releases: bool = True,
    max_gap_onsets: int = 0,
    octave_intervals: Sequence[int] = (12, 24, 36),
    max_streak_pitch_distance: int = 12,
) -> pd.DataFrame:
    """Remove within-instrument octave doublings from a music_df.

    Detects notes doubled at the octave *within* a single instrument
    (e.g., piano playing C4+C5 simultaneously) over consecutive onsets,
    and removes one copy.

    Uses bass-detection to decide which voice to keep: if any note in the
    doubled streak is the lowest sounding note (across all instruments)
    at its onset, the streak is treated as a bass doubling (keep lower,
    drop higher); otherwise keep higher, drop lower.

    Parameters
    ----------
    df : pd.DataFrame
        A music_df with at least ``type``, ``onset``, ``release``, ``pitch``.
    instrument_columns : sequence of str or None
        Columns that jointly identify an instrument. If *None*, auto-detected.
    min_length : int
        Minimum consecutive onsets with octave pairs to count as a doubling.
    quantize : bool
        Round onset/release to a grid before comparison.
    ticks_per_quarter : int
        Grid resolution for onsets when *quantize* is True.
    release_ticks_per_quarter : int or None
        Grid resolution for releases. If *None*, uses *ticks_per_quarter*.
    match_releases : bool
        If True, octave pairs must also have matching (quantized) releases.
    max_gap_onsets : int
        Allow up to this many consecutive onsets without the pair before
        breaking the streak.
    octave_intervals : sequence of int
        Pitch intervals to treat as octave doublings (default: 12, 24, 36).
    max_streak_pitch_distance : int
        Maximum distance (in MIDI semitones) between the lower pitches of
        consecutive pairs for them to belong to the same streak. Prevents
        unrelated octave pairs at the same interval but different registers
        from merging into one streak.

    Returns
    -------
    pd.DataFrame
        The dedoubled dataframe with ``original_index`` column and attrs.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "type":    ["note"]*6,
    ...     "track":   [1]*6,
    ...     "pitch":   [60, 72, 62, 74, 64, 76],
    ...     "onset":   [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    ...     "release": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    ... })
    >>> result = dedouble_octaves_within_instrument(
    ...     df, instrument_columns=["track"])
    >>> result.attrs["n_dedoubled_notes"]
    3
    """
    df = df.copy()
    n_undedoubled_notes = int((df.type == "note").sum())

    note_df, n_non_notes, instrument_columns = _prepare_notes(
        df, instrument_columns, quantize, ticks_per_quarter,
        release_ticks_per_quarter,
    )

    if note_df.empty:
        return _build_output(df, set(), n_undedoubled_notes, n_non_notes)

    # Compute min sounding pitch from ALL instruments combined
    min_sounding = _min_sounding_pitch_at_onsets(
        note_df["_onset_q"].values,
        note_df["_release_q"].values,
        note_df["pitch"].values,
    )
    idx_to_onset_q = dict(zip(note_df.index, note_df["_onset_q"]))
    idx_to_pitch = dict(zip(note_df.index, note_df["pitch"]))

    octave_intervals_set = set(octave_intervals)
    drop_indices: set[int] = set()

    # Integer factor-encode instrument key for faster groupby
    inst_cols = note_df[instrument_columns]
    inst_key_int = np.zeros(len(note_df), dtype=np.int64)
    for col in instrument_columns:
        vals = inst_cols[col].values
        _, codes = np.unique(vals, return_inverse=True)
        inst_key_int = inst_key_int * (codes.max() + 1) + codes
    note_df["_inst_key_int"] = inst_key_int

    for _inst_key, grp in note_df.groupby("_inst_key_int", sort=True):
        grp_sorted = grp.sort_values(["_onset_q", "pitch"])

        # Extract arrays once per instrument group
        pitches = grp_sorted["pitch"].values
        onset_qs = grp_sorted["_onset_q"].values
        release_qs = grp_sorted["_release_q"].values
        orig_indices = grp_sorted.index.values

        unique_onsets = np.unique(onset_qs)
        onset_starts = np.searchsorted(onset_qs, unique_onsets, side="left")
        onset_ends = np.searchsorted(onset_qs, unique_onsets, side="right")

        active_streaks: list[_Streak] = []

        for o_idx in range(len(unique_onsets)):
            s = onset_starts[o_idx]
            e = onset_ends[o_idx]
            n_at = e - s

            # Find all octave pairs at this onset using array slices
            pairs_at_onset: list[tuple[int, int, int, float, float]] = []
            for i_a in range(n_at):
                for i_b in range(i_a + 1, n_at):
                    pa = pitches[s + i_a]
                    pb = pitches[s + i_b]
                    interval = int(abs(pa - pb))
                    if interval not in octave_intervals_set:
                        continue
                    if pa < pb:
                        lo_idx, hi_idx = s + i_a, s + i_b
                        lo_p, hi_p = pa, pb
                    else:
                        lo_idx, hi_idx = s + i_b, s + i_a
                        lo_p, hi_p = pb, pa
                    if match_releases and release_qs[lo_idx] != release_qs[hi_idx]:
                        continue
                    pairs_at_onset.append((
                        interval,
                        int(orig_indices[lo_idx]),
                        int(orig_indices[hi_idx]),
                        lo_p, hi_p,
                    ))

            # Match active streaks to pairs at this onset
            consumed: set[int] = set()
            streaks_to_remove: list[int] = []

            for s_idx, streak in enumerate(active_streaks):
                best_pair_idx = None
                best_dist = float("inf")
                for p_idx, (interval, _li, _ui, lp, _up) in enumerate(
                    pairs_at_onset
                ):
                    if p_idx in consumed:
                        continue
                    if interval != streak.interval:
                        continue
                    dist = abs(lp - streak.last_lower_pitch)
                    if dist > max_streak_pitch_distance:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_pair_idx = p_idx

                if best_pair_idx is not None:
                    consumed.add(best_pair_idx)
                    _, li, ui, lp, up = pairs_at_onset[best_pair_idx]
                    streak.lower_indices.append(li)
                    streak.upper_indices.append(ui)
                    streak.last_lower_pitch = lp
                    streak.length += 1
                    streak.gap = 0
                else:
                    streak.gap += 1
                    if streak.gap > max_gap_onsets:
                        if streak.length >= min_length:
                            _finalize_streak(
                                streak, min_sounding,
                                idx_to_onset_q, idx_to_pitch,
                                drop_indices,
                            )
                        streaks_to_remove.append(s_idx)

            for s_idx in reversed(streaks_to_remove):
                del active_streaks[s_idx]

            # Start new streaks for unconsumed pairs
            for p_idx, (interval, li, ui, lp, up) in enumerate(
                pairs_at_onset
            ):
                if p_idx in consumed:
                    continue
                active_streaks.append(_Streak(
                    interval=interval,
                    lower_indices=[li],
                    upper_indices=[ui],
                    last_lower_pitch=lp,
                    length=1,
                    gap=0,
                ))

        # Finalize remaining streaks
        for streak in active_streaks:
            if streak.length >= min_length:
                _finalize_streak(
                    streak, min_sounding, idx_to_onset_q, idx_to_pitch,
                    drop_indices,
                )

    return _build_output(df, drop_indices, n_undedoubled_notes, n_non_notes)


def _finalize_streak(
    streak: _Streak,
    min_sounding_pitch: dict[float, float],
    idx_to_onset_q: dict[int, float],
    idx_to_pitch: dict[int, float],
    drop_indices: set[int],
) -> None:
    is_bass = False
    for li in streak.lower_indices:
        onset_q = idx_to_onset_q[li]
        pitch = idx_to_pitch[li]
        if pitch <= min_sounding_pitch.get(onset_q, _INF):
            is_bass = True
            break
    if is_bass:
        drop_indices.update(streak.upper_indices)
    else:
        drop_indices.update(streak.lower_indices)


def _build_output(
    df: pd.DataFrame,
    drop_indices: set[int],
    n_undedoubled_notes: int,
    n_non_notes: int,
) -> pd.DataFrame:
    # Drop temp columns if present
    temp_cols = [
        c for c in ("_inst_key", "_inst_key_int", "_onset_q", "_release_q",
                     "inst_key", "onset_q", "release_q")
        if c in df.columns
    ]
    if temp_cols:
        df = df.drop(columns=temp_cols)

    result = df.drop(index=list(drop_indices))

    if "original_index" not in result.columns:
        result = result.reset_index(names="original_index")
    else:
        result = result.reset_index(drop=True)

    n_dedoubled_notes = int((result.type == "note").sum())
    assert (result.type != "note").sum() == n_non_notes, (
        "Non-note row count changed during dedoubling"
    )
    assert n_dedoubled_notes <= n_undedoubled_notes, (
        "Dedoubled notes exceed undedoubled notes"
    )

    result.attrs["n_undedoubled_notes"] = n_undedoubled_notes
    result.attrs["n_dedoubled_notes"] = n_dedoubled_notes
    result.attrs["dedoubled_instruments"] = True

    return result
