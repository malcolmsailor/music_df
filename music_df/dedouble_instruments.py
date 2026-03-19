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

from dataclasses import dataclass, field

from music_df.transforms import transform
from typing import Callable, Sequence

import numpy as np
import pandas as pd

CANDIDATE_INSTRUMENT_COLUMNS = ("instrument", "part", "track", "channel")

DEFAULT_PITCH_THRESHOLD = 53.0  # ~F3


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
        return _build_output(df, set(), n_undedoubled_notes, n_non_notes)

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

    # --- Sort once, then groups are pre-sorted ---
    note_df = note_df.sort_values(["_inst_key", "onset", "pitch"])
    orig_indices = note_df.index.values

    inst_key_vals = note_df["_inst_key"].values
    token_vals = note_df["_token"].values
    pitch_col = note_df["pitch"].values

    # --- Build per-group arrays without iterrows ---
    unique_insts, group_starts = np.unique(inst_key_vals, return_index=True)
    group_ends = np.append(group_starts[1:], len(inst_key_vals))

    sequences: list[np.ndarray] = []
    index_maps: list[np.ndarray] = []
    inst_labels: list[np.ndarray] = []
    pitch_arrays: list[np.ndarray] = []
    sentinel = -1

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

    # --- Concatenate ---
    concatenated = np.concatenate(sequences)
    all_indices = np.concatenate(index_maps)
    all_inst = np.concatenate(inst_labels)
    all_pitches = np.concatenate(pitch_arrays)

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
        return _build_output(df, set(), n_undedoubled_notes, n_non_notes)

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
        drop_inst = drop_selector(inst_a, inst_b, passage_mean_a, passage_mean_b)
        drop_pos = pos_a if drop_inst == inst_a else pos_b

        # Vectorized offset collection
        end = min(drop_pos + match_len, total_len)
        offsets = np.arange(drop_pos, end)
        valid_mask = all_inst[offsets] >= 0
        orig = all_indices[offsets[valid_mask]]
        valid_orig = orig[orig >= 0]
        drop_indices.update(valid_orig.tolist())

    # --- Build output ---
    return _build_output(df, drop_indices, n_undedoubled_notes, n_non_notes)


@transform
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


@transform
def dedouble_octaves(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None = None,
    min_length: int = 3,
    quantize: bool = True,
    ticks_per_quarter: int = 16,
    release_ticks_per_quarter: int | None = None,
    pitch_threshold: float = DEFAULT_PITCH_THRESHOLD,
) -> pd.DataFrame:
    """Remove cross-instrument octave doublings from a music_df.

    Like ``dedouble_unisons_across_instruments`` but matches by pitch class (pitch % 12)
    instead of exact pitch. Defaults to *min_length=3* to reduce false
    positives from contrary motion.

    *pitch_threshold* controls which voice to keep: doublings whose mean pitch
    is >= threshold keep the higher voice (melody), otherwise keep the lower
    voice (bass).

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
    def _drop_by_register(
        inst_a: int, inst_b: int, passage_mean_a: float, passage_mean_b: float
    ) -> int:
        overall_mean = (passage_mean_a + passage_mean_b) / 2
        if overall_mean >= pitch_threshold:
            # Melody register: keep higher, drop lower
            return inst_a if passage_mean_a < passage_mean_b else inst_b
        else:
            # Bass register: keep lower, drop higher
            return inst_a if passage_mean_a > passage_mean_b else inst_b

    return _find_doublings(
        df, instrument_columns, min_length, quantize, ticks_per_quarter,
        pitch_key_mode="mod12",
        drop_selector=_drop_by_register,
        release_ticks_per_quarter=release_ticks_per_quarter,
    )


# ---------------------------------------------------------------------------
# Within-instrument octave dedoubling
# ---------------------------------------------------------------------------

@dataclass
class _Streak:
    """Tracks an active run of octave-doubled onsets within one instrument."""

    interval: int = 0
    lower_indices: list[int] = field(default_factory=list)
    upper_indices: list[int] = field(default_factory=list)
    all_pitches: list[float] = field(default_factory=list)
    last_lower_pitch: float = 0.0
    length: int = 0
    gap: int = 0


@transform
def dedouble_octaves_within_instrument(
    df: pd.DataFrame,
    instrument_columns: Sequence[str] | None = None,
    min_length: int = 3,
    quantize: bool = True,
    ticks_per_quarter: int = 16,
    release_ticks_per_quarter: int | None = None,
    pitch_threshold: float = DEFAULT_PITCH_THRESHOLD,
    match_releases: bool = True,
    max_gap_onsets: int = 0,
    octave_intervals: Sequence[int] = (12, 24, 36),
    max_streak_pitch_distance: int = 12,
) -> pd.DataFrame:
    """Remove within-instrument octave doublings from a music_df.

    Detects notes doubled at the octave *within* a single instrument
    (e.g., piano playing C4+C5 simultaneously) over consecutive onsets,
    and removes one copy.

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
    pitch_threshold : float
        MIDI pitch threshold: doublings above keep the higher voice,
        below keep the lower voice.
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

    octave_intervals_set = set(octave_intervals)
    drop_indices: set[int] = set()

    # Rename helper columns to avoid underscore prefix (itertuples strips it)
    note_df = note_df.rename(columns={
        "_onset_q": "onset_q", "_release_q": "release_q", "_inst_key": "inst_key",
    })

    for _inst_name, grp in note_df.groupby("inst_key", sort=True):
        grp_sorted = grp.sort_values(["onset_q", "pitch"])
        onsets = sorted(grp_sorted["onset_q"].unique())

        active_streaks: list[_Streak] = []

        for onset_q in onsets:
            at_onset = grp_sorted[grp_sorted["onset_q"] == onset_q]
            notes = list(at_onset.itertuples())

            # Find all octave pairs at this onset
            # Each element: (interval, lower_idx, upper_idx, lower_pitch, upper_pitch)
            pairs_at_onset: list[tuple[int, int, int, float, float]] = []
            for i_a in range(len(notes)):
                for i_b in range(i_a + 1, len(notes)):
                    na, nb = notes[i_a], notes[i_b]
                    interval = int(abs(na.pitch - nb.pitch))
                    if interval not in octave_intervals_set:
                        continue
                    lower = na if na.pitch < nb.pitch else nb
                    upper = nb if na.pitch < nb.pitch else na
                    if match_releases and lower.release_q != upper.release_q:
                        continue
                    pairs_at_onset.append((
                        interval, lower.Index, upper.Index,
                        lower.pitch, upper.pitch,
                    ))

            # Match active streaks to pairs at this onset
            consumed: set[int] = set()  # indices into pairs_at_onset
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
                    streak.all_pitches.extend([lp, up])
                    streak.last_lower_pitch = lp
                    streak.length += 1
                    streak.gap = 0
                else:
                    streak.gap += 1
                    if streak.gap > max_gap_onsets:
                        if streak.length >= min_length:
                            _finalize_streak(
                                streak, pitch_threshold, drop_indices
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
                    all_pitches=[lp, up],
                    last_lower_pitch=lp,
                    length=1,
                    gap=0,
                ))

        # Finalize remaining streaks
        for streak in active_streaks:
            if streak.length >= min_length:
                _finalize_streak(streak, pitch_threshold, drop_indices)

    return _build_output(df, drop_indices, n_undedoubled_notes, n_non_notes)


def _finalize_streak(
    streak: _Streak,
    pitch_threshold: float,
    drop_indices: set[int],
) -> None:
    mean_pitch = np.mean(streak.all_pitches)
    if mean_pitch >= pitch_threshold:
        # Melody register: keep higher, drop lower
        drop_indices.update(streak.lower_indices)
    else:
        # Bass register: keep lower, drop higher
        drop_indices.update(streak.upper_indices)


def _build_output(
    df: pd.DataFrame,
    drop_indices: set[int],
    n_undedoubled_notes: int,
    n_non_notes: int,
) -> pd.DataFrame:
    # Drop temp columns if present
    temp_cols = [
        c for c in ("_inst_key", "_onset_q", "_release_q",
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
