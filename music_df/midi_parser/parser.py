"""
MIDI parsing using symusic.

This module previously used mido for MIDI parsing. It now uses symusic,
which is faster and handles edge cases (overlapping notes, etc.) internally.
"""

import csv
import fractions
import os
import re
import warnings
from typing import Optional, Tuple, Type, Union

import pandas as pd
import symusic

from music_df.conversions.symusic_conv import (
    df_to_symusic_score,
    symusic_score_to_df,
)


class MidiError(Exception):
    pass


def midi_to_table(
    in_midi_fname,
    time_type: Type = float,
    max_denominator: int = 8192,
    overlapping_notes: str = "end_all",
    pb_tup_dict: Optional[dict] = None,
    display_name: str | None = None,
    notes_only: bool = False,
    warn_for_orphan_note_offs: bool = False,
    warn_for_orphan_note_ons: bool = False,
    warn_for_overlapping_notes: bool = False,
) -> pd.DataFrame:
    """Read a midi file and return a pandas DataFrame.

    Note-on and note-off events will be compiled into a single event with
    attack and release.

    Args:
        in_midi_fname: path to input midi file.

    Keyword args:
        time_type: the numeric type that should be used to express time
            attributes. The default is `float`, but if preserving exact relative
            timings is important it may be better to use `fractions.Fraction`
            to avoid float rounding issues. If `int`, then returned in ticks per beat.
        max_denominator: integer. Only has an effect if `time_type` is
            `fractions.Fraction`, in which case this argument sets the
            maximum denominator.
            Default: 8192
        overlapping_notes: DEPRECATED - symusic handles overlapping notes internally.
            This parameter is ignored.
        pb_tup_dict: DEPRECATED - Custom pitch-bend mapping is no longer supported.
            This parameter is ignored.
        display_name: the value of the "filename" column in the returned dataframe. If
            not passed, uses in_midi_fname.
        notes_only: If True, only include note events.
        warn_for_orphan_note_offs: DEPRECATED - symusic handles this internally.
            This parameter is ignored.
        warn_for_orphan_note_ons: DEPRECATED - symusic handles this internally.
            This parameter is ignored.
        warn_for_overlapping_notes: DEPRECATED - symusic handles this internally.
            This parameter is ignored.

    Returns: a dataframe "events".
        - note events are the combination of a note-on with the following
            note-off message.
        - all other midi messages map to a single event.

        All events have onset, release, channel, pitch, velocity,
        and 'other' fields. pitch, velocity, and duration are null
        for non-note events; channel is null for events that do not have a
        channel attribute; all other fields go into "other" as a string
        representation.

        Tracks and channels are zero-indexed.

        Output will be sorted with `sort_df()` function.

    Raises:
        MidiError: If the file cannot be read.

    Note:
        symusic creates Note objects by pairing note_on with note_off events.
        MIDI files with unpaired note_ons (no corresponding note_off) will have
        those notes silently dropped. This can occur with some percussion-only
        files where drum hits are treated as one-shots.
    """
    if pb_tup_dict is not None:
        warnings.warn(
            "pb_tup_dict parameter is deprecated and ignored. "
            "symusic does not support custom pitch-bend mapping.",
            DeprecationWarning,
            stacklevel=2,
        )

    if overlapping_notes != "end_all":
        warnings.warn(
            "overlapping_notes parameter is deprecated and ignored. "
            "symusic handles overlapping notes internally (behavior matches 'end_all').",
            DeprecationWarning,
            stacklevel=2,
        )

    if display_name is None:
        display_name = os.path.basename(in_midi_fname)

    try:
        score = symusic.Score(in_midi_fname)
    except Exception as exc:
        raise MidiError(f"unable to read file {in_midi_fname}") from exc

    return symusic_score_to_df(
        score,
        time_type=time_type,
        max_denominator=max_denominator,
        display_name=display_name,
        notes_only=notes_only,
    )


def midi_to_csv(in_midi_fname, out_csv_fname, *args, **kwargs):
    df = midi_to_table(in_midi_fname, *args, **kwargs)
    df.to_csv(out_csv_fname, index=False)


def midi_dirs_to_csv(list_of_dirs, out_csv_fname):
    raise NotImplementedError("I need to update to pandas version of midi_to_table")


def midi_files_to_csv(list_of_files, out_csv_fname, append=False):
    raise NotImplementedError("I need to update to pandas version of midi_to_table")


def df_to_midi(
    df: pd.DataFrame,
    midi_path: str,
    ts: Optional[Union[str, Tuple[int, int]]] = None,
    ticks_per_quarter: int = 480,
) -> None:
    """Convert a music_df DataFrame to a MIDI file.

    Args:
        df: A music_df DataFrame with columns: type, onset, release, track, pitch,
            velocity, other.
        midi_path: Output path for the MIDI file.
        ts: Optional time signature to add. Can be a string like "4/4" or a tuple
            like (4, 4). If None, no time signature is added unless one exists in df.
        ticks_per_quarter: Ticks per quarter note.
    """
    score = df_to_symusic_score(df, ticks_per_quarter=ticks_per_quarter)

    # If ts is provided, add it to the score
    if ts is not None:
        if isinstance(ts, str):
            m = re.match(r"(?P<numer>\d+)/(?P<denom>\d+)$", ts)
            assert m is not None, f"Invalid time signature format: {ts}"
            numer = int(m.group("numer"))
            denom = int(m.group("denom"))
        else:
            numer, denom = ts

        # Check if there's already a time signature at time 0
        has_ts_at_zero = any(ts_event.time == 0 for ts_event in score.time_signatures)

        if not has_ts_at_zero:
            score.time_signatures.append(
                symusic.TimeSignature(time=0, numerator=numer, denominator=denom)
            )

    score.dump_midi(midi_path)
