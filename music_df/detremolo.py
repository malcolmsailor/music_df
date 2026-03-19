"""
Provides functions for merging repeated notes in a dataframe.
"""

from typing import Iterable

import numpy as np
import pandas as pd

from music_df.transforms import transform
from music_df.transpose import PERCUSSION_CHANNEL

_EPSILON = 1e-6


def _compute_outer_voice_flags(
    notes: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return (is_bass, is_soprano) boolean Series for each note.

    A note is "bass" if it has the lowest pitch among all non-percussion notes
    sounding at its onset; "soprano" if it has the highest.
    """
    if "channel" in notes.columns:
        cand_mask = notes["channel"].values != PERCUSSION_CHANNEL
    else:
        cand_mask = np.ones(len(notes), dtype=bool)

    cand_onsets = notes["onset"].values[cand_mask]
    cand_releases = notes["release"].values[cand_mask]
    cand_pitches = notes["pitch"].values[cand_mask]

    if len(cand_onsets) == 0:
        return (
            pd.Series(False, index=notes.index),
            pd.Series(False, index=notes.index),
        )

    note_onsets = notes["onset"].values
    note_pitches = notes["pitch"].values
    unique_onsets = np.unique(note_onsets)
    n_unique = len(unique_onsets)

    # For each unique onset, find min/max pitch among sounding candidates.
    # Iterate over distinct pitches: for each pitch p, use searchsorted to
    # count how many candidates with that pitch are active at each onset.
    # This is O(P * (N log N + U)) — much faster than O(U * C).
    unique_pitches = np.unique(cand_pitches).astype(np.float64)
    n_pitches = len(unique_pitches)

    # is_active[pi, ui]: whether any candidate with unique_pitches[pi] is
    # sounding at unique_onsets[ui]
    is_active = np.zeros((n_pitches, n_unique), dtype=bool)
    for pi, p in enumerate(unique_pitches):
        mask = cand_pitches == p
        starts_p = np.sort(cand_onsets[mask])
        ends_p = np.sort(cand_releases[mask])
        started = np.searchsorted(starts_p, unique_onsets, side="right")
        ended = np.searchsorted(ends_p, unique_onsets, side="right")
        is_active[pi] = (started - ended) > 0

    pitch_col = unique_pitches[:, np.newaxis]
    _SENTINEL_HIGH = 999.0
    _SENTINEL_LOW = -1.0
    min_at_onset = np.where(is_active, pitch_col, _SENTINEL_HIGH).min(axis=0)
    max_at_onset = np.where(is_active, pitch_col, _SENTINEL_LOW).max(axis=0)

    any_active = is_active.any(axis=0)
    min_at_onset[~any_active] = np.inf
    max_at_onset[~any_active] = -np.inf

    note_onset_idx = np.searchsorted(unique_onsets, note_onsets)
    is_bass = (note_pitches == min_at_onset[note_onset_idx]) & cand_mask
    is_soprano = (note_pitches == max_at_onset[note_onset_idx]) & cand_mask

    return (
        pd.Series(is_bass, index=notes.index),
        pd.Series(is_soprano, index=notes.index),
    )


def _build_instr_key(notes: pd.DataFrame, group_cols: list[str]) -> np.ndarray:
    """Build a single integer key encoding the instrument group for each note."""
    if not group_cols:
        return np.zeros(len(notes), dtype=np.int64)
    # Encode each column as a factor, combine into a single key
    key = np.zeros(len(notes), dtype=np.int64)
    multiplier = 1
    for col in group_cols:
        vals = notes[col].values
        _, codes = np.unique(vals, return_inverse=True)
        key += codes * multiplier
        multiplier *= len(np.unique(vals)) + 1
    return key


@transform
def merge_repeated_notes(
    df: pd.DataFrame,
    max_note_duration: float | None = None,
    max_gap: float = 0.125,
    instrument_columns: Iterable[str] = (
        "instrument",
        "midi_instrument",
        "track",
        "channel",
    ),
    preserve_outer_voices: bool = True,
) -> pd.DataFrame:
    """
    Merge repeated notes of the same pitch into single notes.

    Args:
        df: dataframe.
        max_note_duration: maximum duration of a note to be eligible for merging.
            If None, all notes are eligible regardless of duration.
        max_gap: maximum gap between the release of one note and the onset of
            the next note of the same pitch for them to be merged.
        instrument_columns: columns to group by. Only notes that share the same
            values in these columns can be merged.
        preserve_outer_voices: if True, don't merge across changes in
            bass/soprano status (i.e., whether the note is the lowest or
            highest sounding pitch at its onset).
    """
    notes = df[df["type"] == "note"]
    if len(notes) < 2:
        return df.copy()

    group_cols = [c for c in instrument_columns if c in df.columns]

    if preserve_outer_voices:
        is_bass_s, is_soprano_s = _compute_outer_voice_flags(notes)
        bass_vals = is_bass_s.values
        soprano_vals = is_soprano_s.values
    else:
        bass_vals = soprano_vals = None

    # Build a composite key: (instrument_group, pitch)
    instr_key = _build_instr_key(notes, group_cols)
    pitches = notes["pitch"].values
    onsets = notes["onset"].values
    releases = notes["release"].values
    orig_idx = notes.index.values

    # Sort by (instr_key, pitch, onset) to put same-group same-pitch notes
    # adjacent, ordered by onset
    sort_order = np.lexsort((onsets, pitches, instr_key))
    s_instr = instr_key[sort_order]
    s_pitch = pitches[sort_order]
    s_onset = onsets[sort_order]
    s_release = releases[sort_order]
    s_orig_idx = orig_idx[sort_order]
    if bass_vals is not None:
        s_bass = bass_vals[sort_order]
        s_soprano = soprano_vals[sort_order]

    n = len(sort_order)

    # Identify boundaries between (instr, pitch) groups
    same_group = (s_instr[:-1] == s_instr[1:]) & (s_pitch[:-1] == s_pitch[1:])

    # gap_to_next: onset[i+1] - release[i], only meaningful within same group
    gap = s_onset[1:] - s_release[:-1]

    near_zero = gap <= _EPSILON
    within_max = gap <= max_gap

    # has_intervening: are there any onsets (of ANY pitch in the same
    # instrument group) between release[i] and onset[i+1]?
    # We need the sorted all_onsets per instrument group. To avoid a Python
    # loop, we compute this per instrument group.
    # Build sorted onset arrays per instrument group using numpy
    unique_instr_keys = np.unique(instr_key)
    # Map each note (in sort_order) to its position for has_intervening check
    has_interv = np.zeros(n - 1, dtype=bool)

    # Process instrument groups — this loop is over instrument groups (few),
    # not over pitch groups (many)
    for ik in unique_instr_keys:
        # Mask for notes in this instrument group (in sorted order)
        instr_mask = s_instr == ik
        # All onsets in this instrument group (sorted)
        instr_onsets_all = np.sort(onsets[instr_key == ik])

        # Pairs within this group: positions where same_group is True AND
        # both notes are in this instrument group
        pair_mask = same_group & instr_mask[:-1]
        if not pair_mask.any():
            continue

        pair_indices = np.where(pair_mask)[0]
        rel = s_release[pair_indices]
        nxt = s_onset[pair_indices + 1]

        release_pos = np.searchsorted(instr_onsets_all, rel, side="left")
        next_onset_pos = np.searchsorted(instr_onsets_all, nxt, side="left")
        has_interv[pair_indices] = release_pos < next_onset_pos

    # can_continue[i]: whether note i+1 can merge with note i
    can_continue = same_group & (
        near_zero | (within_max & ~has_interv)
    )

    if max_note_duration is not None:
        duration = s_release[:-1] - s_onset[:-1]
        can_continue = can_continue & (duration <= max_note_duration)

    if bass_vals is not None:
        same_voice = (s_bass[:-1] == s_bass[1:]) & (s_soprano[:-1] == s_soprano[1:])
        can_continue = can_continue & same_voice

    # Build merge group IDs: a new group starts where can_continue is False
    starts_new = np.empty(n, dtype=bool)
    starts_new[0] = True
    starts_new[1:] = ~can_continue

    merge_ids = np.cumsum(starts_new)

    # Find first/last in each merge group and group sizes
    is_first = np.empty(n, dtype=bool)
    is_first[0] = True
    is_first[1:] = merge_ids[1:] != merge_ids[:-1]

    is_last = np.empty(n, dtype=bool)
    is_last[-1] = True
    is_last[:-1] = merge_ids[:-1] != merge_ids[1:]

    group_sizes = np.bincount(merge_ids)
    multi = group_sizes[merge_ids] > 1

    # First notes of multi-note groups get the release of the last note
    first_of_multi = is_first & multi
    last_of_multi = is_last & multi
    release_update_indices = s_orig_idx[first_of_multi]
    release_update_values = s_release[last_of_multi]

    # Non-first notes of multi-note groups are dropped
    to_drop = s_orig_idx[~is_first & multi]

    out_df = df.copy()
    if len(release_update_indices) > 0:
        out_df.loc[release_update_indices, "release"] = release_update_values

    return out_df.drop(to_drop, axis=0)


@transform
def detremolo(
    df: pd.DataFrame,
    max_tremolo_note_length: float = 0.25,
    max_tremolo_note_gap: float = 0.125,
    instrument_columns: Iterable[str] = (
        "instrument",
        "midi_instrument",
        "track",
        "channel",
    ),
    preserve_outer_voices: bool = True,
) -> pd.DataFrame:
    """
    Merge rapid repeated notes (tremolo) into single notes.

    Args:
        df: dataframe.
        max_tremolo_note_length: maximum length of a note to be eligible for
            a tremolo.
        max_tremolo_note_gap: maximum gap between the release of one note and
            the onset of the next note of the same pitch to be eligible for
            a tremolo.
        instrument_columns: columns to group by when detremoloing. This permits
            us to only merge notes that seem to belong to the same instrument.
        preserve_outer_voices: if True, don't merge across changes in
            bass/soprano status.
    """
    return merge_repeated_notes(
        df,
        max_note_duration=max_tremolo_note_length,
        max_gap=max_tremolo_note_gap,
        instrument_columns=instrument_columns,
        preserve_outer_voices=preserve_outer_voices,
    )
