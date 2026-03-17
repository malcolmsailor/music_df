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
    is_bass = pd.Series(False, index=notes.index)
    is_soprano = pd.Series(False, index=notes.index)

    if "channel" in notes.columns:
        candidates = notes[notes["channel"] != PERCUSSION_CHANNEL]
    else:
        candidates = notes

    if len(candidates) == 0:
        return is_bass, is_soprano

    cand_onsets = candidates["onset"].values
    cand_releases = candidates["release"].values
    cand_pitches = candidates["pitch"].values
    cand_indices = candidates.index.values

    for t in np.unique(notes["onset"].values):
        sounding = (cand_onsets <= t) & (cand_releases > t)
        if not sounding.any():
            continue
        sounding_pitches = cand_pitches[sounding]
        sounding_indices = cand_indices[sounding]

        min_pitch = sounding_pitches.min()
        max_pitch = sounding_pitches.max()

        # All notes sounding at this onset that have this onset get flagged
        onset_mask = notes.index.isin(
            sounding_indices[sounding_pitches == min_pitch]
        ) & (notes["onset"] == t)
        is_bass = is_bass | onset_mask

        onset_mask = notes.index.isin(
            sounding_indices[sounding_pitches == max_pitch]
        ) & (notes["onset"] == t)
        is_soprano = is_soprano | onset_mask

    return is_bass, is_soprano


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
    group_cols = [c for c in instrument_columns if c in df.columns]

    if preserve_outer_voices and len(notes) > 0:
        is_bass, is_soprano = _compute_outer_voice_flags(notes)
    else:
        is_bass = is_soprano = None

    to_drop = []
    release_updates = {}

    instr_groups = (
        notes.groupby(group_cols) if group_cols else [("_all", notes)]
    )
    for _, instr in instr_groups:
        all_onsets = np.sort(instr["onset"].values)
        for _, group in instr.groupby("pitch"):
            if len(group) < 2:
                continue

            duration = group["release"] - group["onset"]
            gap_to_next = group["onset"].shift(-1) - group["release"]

            near_zero_gap = gap_to_next <= _EPSILON
            within_max_gap = gap_to_next <= max_gap

            releases = group["release"].values
            next_onsets = group["onset"].shift(-1).values
            has_intervening = np.array([
                (
                    False
                    if np.isnan(next_onsets[i])
                    else np.searchsorted(all_onsets, releases[i], side="left")
                    < np.searchsorted(all_onsets, next_onsets[i], side="left")
                )
                for i in range(len(group))
            ])
            has_intervening_s = pd.Series(
                has_intervening, index=group.index
            )

            can_continue = near_zero_gap | (
                within_max_gap & ~has_intervening_s
            )
            if max_note_duration is not None:
                can_continue = can_continue & (duration <= max_note_duration)
            if is_bass is not None:
                same_voice_role = (
                    is_bass.loc[group.index]
                    == is_bass.loc[group.index].shift(-1)
                ) & (
                    is_soprano.loc[group.index]
                    == is_soprano.loc[group.index].shift(-1)
                )
                # Last note in group gets NaN from shift; fill True so it
                # doesn't spuriously break a chain
                same_voice_role = same_voice_role.fillna(True)
                can_continue = can_continue & same_voice_role

            starts_new_group = ~can_continue.shift(1, fill_value=False)
            group_ids = starts_new_group.cumsum()

            for _, merge_group in group.groupby(group_ids):
                if len(merge_group) > 1:
                    release_updates[merge_group.index[0]] = merge_group.iloc[
                        -1
                    ]["release"]
                    to_drop.extend(merge_group.index[1:])

    out_df = df.copy()
    for idx, release in release_updates.items():
        out_df.loc[idx, "release"] = release

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
