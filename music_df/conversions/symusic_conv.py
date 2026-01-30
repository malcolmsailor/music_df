"""
Functions for converting between symusic Score objects and music_df DataFrames.
"""

import fractions
from typing import Type

import pandas as pd
import symusic

from music_df.sort_df import sort_df


def symusic_score_to_df(
    score: symusic.Score,
    time_type: Type = float,
    max_denominator: int = 8192,
    display_name: str | None = None,
    notes_only: bool = False,
) -> pd.DataFrame:
    """Convert a symusic Score to a music_df DataFrame.

    Args:
        score: A symusic Score object.
        time_type: The numeric type for time values. Default is float.
            Use fractions.Fraction for exact timing, or int for raw ticks.
        max_denominator: Maximum denominator when time_type is fractions.Fraction.
        display_name: Value for the "filename" column. If None, uses empty string.
        notes_only: If True, only include note events (no tempos, time signatures, etc.).

    Returns:
        A DataFrame with columns: filename, type, onset, release, track, channel,
        pitch, velocity, other.

    Note:
        symusic creates Note objects by pairing note_on with note_off events.
        MIDI files with unpaired note_ons (no corresponding note_off) will have
        those notes silently dropped. This can occur with some percussion-only
        files where drum hits are treated as one-shots.
    """
    if display_name is None:
        display_name = ""

    ticks_per_quarter = score.ticks_per_quarter

    def _get_time(tick_time: int):
        if time_type == int:
            return tick_time
        if time_type == fractions.Fraction:
            return fractions.Fraction(tick_time, ticks_per_quarter).limit_denominator(
                max_denominator=max_denominator
            )
        return time_type(tick_time / ticks_per_quarter)

    rows: list[dict] = []

    # Extract notes from all tracks
    for track_i, track in enumerate(score.tracks):
        channel = 9 if track.is_drum else 0
        for note in track.notes:
            rows.append(
                {
                    "filename": display_name,
                    "type": "note",
                    "onset": _get_time(note.time),
                    "release": _get_time(note.time + note.duration),
                    "track": track_i,
                    "channel": channel,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "other": None,
                }
            )

    if not notes_only:
        # Extract tempo events
        for tempo in score.tempos:
            rows.append(
                {
                    "filename": display_name,
                    "type": "tempo",
                    "onset": _get_time(tempo.time),
                    "release": None,
                    "track": None,
                    "channel": None,
                    "pitch": None,
                    "velocity": None,
                    "other": {"tempo": tempo.qpm},
                }
            )

        # Extract time signature events
        for ts in score.time_signatures:
            if ts.denominator <= 0:
                raise ValueError(
                    f"Malformed MIDI file: invalid time signature denominator ({ts.denominator}). "
                    f"Time signature numerator was {ts.numerator}."
                )
            rows.append(
                {
                    "filename": display_name,
                    "type": "time_signature",
                    "onset": _get_time(ts.time),
                    "release": None,
                    "track": None,
                    "channel": None,
                    "pitch": None,
                    "velocity": None,
                    "other": {"numerator": ts.numerator, "denominator": ts.denominator},
                }
            )

        # Extract key signature events
        for ks in score.key_signatures:
            rows.append(
                {
                    "filename": display_name,
                    "type": "key_signature",
                    "onset": _get_time(ks.time),
                    "release": None,
                    "track": None,
                    "channel": None,
                    "pitch": None,
                    "velocity": None,
                    "other": {"key": ks.key, "tonality": ks.tonality},
                }
            )

        # Extract markers
        for marker in score.markers:
            rows.append(
                {
                    "filename": display_name,
                    "type": "marker",
                    "onset": _get_time(marker.time),
                    "release": None,
                    "track": None,
                    "channel": None,
                    "pitch": None,
                    "velocity": None,
                    "other": {"text": marker.text},
                }
            )

        # Extract track-level events (program changes at onset 0)
        for track_i, track in enumerate(score.tracks):
            if track.program is not None:
                rows.append(
                    {
                        "filename": display_name,
                        "type": "program_change",
                        "onset": _get_time(0),
                        "release": None,
                        "track": track_i,
                        "channel": 0,
                        "pitch": None,
                        "velocity": None,
                        "other": {"program": track.program},
                    }
                )

            # Extract control changes
            for cc in track.controls:
                rows.append(
                    {
                        "filename": display_name,
                        "type": "control_change",
                        "onset": _get_time(cc.time),
                        "release": None,
                        "track": track_i,
                        "channel": 0,
                        "pitch": None,
                        "velocity": None,
                        "other": {"control": cc.number, "value": cc.value},
                    }
                )

            # Extract pitch bends
            for pb in track.pitch_bends:
                rows.append(
                    {
                        "filename": display_name,
                        "type": "pitchwheel",
                        "onset": _get_time(pb.time),
                        "release": None,
                        "track": track_i,
                        "channel": 0,
                        "pitch": None,
                        "velocity": None,
                        "other": {"pitch": pb.value},
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "filename",
                "type",
                "onset",
                "release",
                "track",
                "channel",
                "pitch",
                "velocity",
                "other",
            ]
        )

    df = pd.DataFrame(rows)
    return sort_df(df)


def df_to_symusic_score(
    df: pd.DataFrame,
    ticks_per_quarter: int = 480,
) -> symusic.Score:
    """Convert a music_df DataFrame to a symusic Score.

    Args:
        df: A music_df DataFrame with columns: type, onset, release, track, pitch,
            velocity, other.
        ticks_per_quarter: Ticks per quarter note for the output Score.

    Returns:
        A symusic Score object.
    """
    score = symusic.Score(ticks_per_quarter)

    # Determine time conversion factor based on df time format
    # Assume times are in quarter notes (beats)
    def _to_ticks(time_val) -> int:
        if time_val is None or pd.isna(time_val):
            return 0
        if isinstance(time_val, fractions.Fraction):
            return int(time_val * ticks_per_quarter)
        return int(float(time_val) * ticks_per_quarter)

    # Determine number of tracks needed
    if "track" in df.columns:
        max_track = df["track"].dropna().max()
        if pd.isna(max_track):
            n_tracks = 1
        else:
            n_tracks = int(max_track) + 1
    else:
        n_tracks = 1

    # Create tracks
    for _ in range(n_tracks):
        score.tracks.append(symusic.Track())

    # Process rows
    for _, row in df.iterrows():
        event_type = row.get("type", "note")

        if event_type == "note":
            track_i = int(row["track"]) if pd.notna(row.get("track")) else 0
            onset_ticks = _to_ticks(row["onset"])
            release_ticks = _to_ticks(row["release"])
            duration_ticks = release_ticks - onset_ticks

            note = symusic.Note(
                time=onset_ticks,
                duration=duration_ticks,
                pitch=int(row["pitch"]),
                velocity=int(row.get("velocity", 64)) if pd.notna(row.get("velocity")) else 64,
            )
            score.tracks[track_i].notes.append(note)

        elif event_type == "tempo":
            tempo_val = row.get("tempo")
            if tempo_val is None and isinstance(row.get("other"), dict):
                tempo_val = row["other"].get("tempo")
            if tempo_val is not None:
                score.tempos.append(
                    symusic.Tempo(time=_to_ticks(row["onset"]), qpm=float(tempo_val))
                )

        elif event_type == "time_signature":
            other = row.get("other", {})
            if isinstance(other, dict):
                score.time_signatures.append(
                    symusic.TimeSignature(
                        time=_to_ticks(row["onset"]),
                        numerator=int(other.get("numerator", 4)),
                        denominator=int(other.get("denominator", 4)),
                    )
                )

        elif event_type == "key_signature":
            other = row.get("other", {})
            if isinstance(other, dict):
                score.key_signatures.append(
                    symusic.KeySignature(
                        time=_to_ticks(row["onset"]),
                        key=int(other.get("key", 0)),
                        tonality=int(other.get("tonality", 0)),
                    )
                )

        elif event_type == "program_change":
            track_i = int(row["track"]) if pd.notna(row.get("track")) else 0
            other = row.get("other", {})
            if isinstance(other, dict) and "program" in other:
                score.tracks[track_i].program = int(other["program"])

        elif event_type == "control_change":
            track_i = int(row["track"]) if pd.notna(row.get("track")) else 0
            other = row.get("other", {})
            if isinstance(other, dict):
                score.tracks[track_i].controls.append(
                    symusic.ControlChange(
                        time=_to_ticks(row["onset"]),
                        number=int(other.get("control", 0)),
                        value=int(other.get("value", 0)),
                    )
                )

        elif event_type == "pitchwheel":
            track_i = int(row["track"]) if pd.notna(row.get("track")) else 0
            other = row.get("other", {})
            if isinstance(other, dict) and "pitch" in other:
                score.tracks[track_i].pitch_bends.append(
                    symusic.PitchBend(
                        time=_to_ticks(row["onset"]),
                        value=int(other["pitch"]),
                    )
                )

    return score


def read_midi_symusic(
    midi_path: str,
    time_type: Type = float,
    max_denominator: int = 8192,
    display_name: str | None = None,
    notes_only: bool = False,
) -> pd.DataFrame:
    """Read a MIDI file and return a music_df DataFrame using symusic.

    This is an alternative to midi_to_table() that uses symusic instead of mido.

    Args:
        midi_path: Path to the MIDI file.
        time_type: The numeric type for time values. Default is float.
            Use fractions.Fraction for exact timing, or int for raw ticks.
        max_denominator: Maximum denominator when time_type is fractions.Fraction.
        display_name: Value for the "filename" column. If None, uses the basename.
        notes_only: If True, only include note events.

    Returns:
        A DataFrame with columns: filename, type, onset, release, track, channel,
        pitch, velocity, other.

    Note:
        symusic creates Note objects by pairing note_on with note_off events.
        MIDI files with unpaired note_ons (no corresponding note_off) will have
        those notes silently dropped. This can occur with some percussion-only
        files where drum hits are treated as one-shots.
    """
    import os

    if display_name is None:
        display_name = os.path.basename(midi_path)

    score = symusic.Score(midi_path)
    return symusic_score_to_df(
        score,
        time_type=time_type,
        max_denominator=max_denominator,
        display_name=display_name,
        notes_only=notes_only,
    )


def write_midi_symusic(
    df: pd.DataFrame,
    midi_path: str,
    ticks_per_quarter: int = 480,
) -> None:
    """Write a music_df DataFrame to a MIDI file using symusic.

    Args:
        df: A music_df DataFrame.
        midi_path: Output path for the MIDI file.
        ticks_per_quarter: Ticks per quarter note.
    """
    score = df_to_symusic_score(df, ticks_per_quarter=ticks_per_quarter)
    score.dump_midi(midi_path)
