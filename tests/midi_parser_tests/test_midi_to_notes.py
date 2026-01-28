import fractions
import math
import os
import tempfile
import warnings

import pandas as pd
import pytest
import symusic

from music_df import sort_df
from music_df.midi_parser import df_to_midi, midi_to_csv, midi_to_table

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))

PALMID = os.path.join(SCRIPT_DIR, "test_files", "misc_Palestrina.mid")

warnings.simplefilter("error")


def create_test_midi(notes: list[dict], ticks_per_quarter: int = 480) -> str:
    """Create a temporary MIDI file with the given notes.

    Args:
        notes: List of dicts with keys: pitch, time (onset in ticks), duration (in ticks),
               velocity (optional, defaults to 64), track (optional, defaults to 0)
        ticks_per_quarter: Ticks per quarter note.

    Returns:
        Path to temporary MIDI file.
    """
    score = symusic.Score(ticks_per_quarter)

    # Determine number of tracks needed
    max_track = max((n.get("track", 0) for n in notes), default=0)
    for _ in range(max_track + 1):
        score.tracks.append(symusic.Track())

    for note in notes:
        track_i = note.get("track", 0)
        score.tracks[track_i].notes.append(
            symusic.Note(
                time=note["time"],
                duration=note["duration"],
                pitch=note["pitch"],
                velocity=note.get("velocity", 64),
            )
        )

    fd, path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    score.dump_midi(path)
    return path


def test_basic_note():
    """Test parsing a single note."""
    path = create_test_midi([{"pitch": 60, "time": 0, "duration": 480}])
    try:
        df = midi_to_table(path, time_type=float)
        notes = df[df["type"] == "note"]
        assert len(notes) == 1
        note = notes.iloc[0]
        assert note["pitch"] == 60
        assert note["onset"] == 0.0
        assert note["release"] == 1.0
    finally:
        os.remove(path)


def test_consecutive_notes():
    """Test parsing consecutive notes on the same pitch."""
    path = create_test_midi(
        [
            {"pitch": 60, "time": 0, "duration": 480},
            {"pitch": 60, "time": 480, "duration": 480},
            {"pitch": 60, "time": 960, "duration": 480},
        ]
    )
    try:
        df = midi_to_table(path, time_type=float)
        notes = df[df["type"] == "note"]
        assert len(notes) == 3
    finally:
        os.remove(path)


def test_overlapping_notes():
    """Test parsing overlapping notes (symusic handles this internally)."""
    path = create_test_midi(
        [
            {"pitch": 60, "time": 0, "duration": 960},
            {"pitch": 60, "time": 480, "duration": 480},
        ]
    )
    try:
        df = midi_to_table(path, time_type=float)
        notes = df[df["type"] == "note"]
        assert len(notes) == 2
    finally:
        os.remove(path)


def test_midi_to_table():
    """Test that midi_to_table works on a real MIDI file."""
    result = midi_to_table(PALMID)
    assert len(result) > 0
    assert "note" in result["type"].values


def test_midi_to_csv():
    """Test midi_to_csv function."""
    _, csv_path = tempfile.mkstemp(suffix=".csv")
    try:
        midi_to_csv(PALMID, csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) > 0
    finally:
        os.remove(csv_path)


def test_df_to_midi():
    """Test round-trip: MIDI -> DataFrame -> MIDI -> DataFrame."""
    orig_df = midi_to_table(PALMID)
    _, csv_path = tempfile.mkstemp(suffix=".csv")
    orig_df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    os.remove(csv_path)

    # "track" is a float if any items are nan; that can cause issues so we
    # explicitly set it to nan here
    df["track"] = df.track.astype(float)

    _, mid_path = tempfile.mkstemp(suffix=".mid")
    df_to_midi(df, mid_path)
    df2 = midi_to_table(mid_path)
    os.remove(mid_path)

    df = df[df.type == "note"].reset_index(drop=True)
    df2 = df2[df2.type == "note"].reset_index(drop=True)
    df.drop(
        columns=["filename", "other", "type", "instrument", "label"],
        inplace=True,
        errors="ignore",
    )
    df2.drop(columns=["filename", "other", "type"], inplace=True, errors="ignore")
    df = sort_df(df)
    df2 = sort_df(df2)

    assert len(df) == len(df2)
    for (_, note1), (_, note2) in zip(df.iterrows(), df2.iterrows()):
        for name, val in note1.items():
            if isinstance(val, float):
                assert math.isclose(val, note2[name])
            else:
                assert val == note2[name]


def test_time_type_float():
    """Test time_type=float."""
    df = midi_to_table(PALMID, time_type=float)
    notes = df[df["type"] == "note"]
    assert all(isinstance(t, float) for t in notes["onset"])


def test_time_type_fraction():
    """Test time_type=fractions.Fraction."""
    df = midi_to_table(PALMID, time_type=fractions.Fraction)
    notes = df[df["type"] == "note"]
    assert all(isinstance(t, fractions.Fraction) for t in notes["onset"])


def test_time_type_int():
    """Test time_type=int (ticks)."""
    df = midi_to_table(PALMID, time_type=int)
    notes = df[df["type"] == "note"]
    assert all(isinstance(t, int) for t in notes["onset"])


def test_notes_only():
    """Test notes_only parameter."""
    df_all = midi_to_table(PALMID, notes_only=False)
    df_notes = midi_to_table(PALMID, notes_only=True)

    assert set(df_notes["type"].unique()) == {"note"}
    assert len(df_all) >= len(df_notes)


def test_display_name():
    """Test display_name parameter."""
    custom_name = "custom_name.mid"
    df = midi_to_table(PALMID, display_name=custom_name)
    assert (df["filename"] == custom_name).all()


def test_deprecated_overlapping_notes_warning():
    """Test that using non-default overlapping_notes emits deprecation warning."""
    with pytest.warns(DeprecationWarning, match="overlapping_notes"):
        midi_to_table(PALMID, overlapping_notes="end_first")


def test_deprecated_pb_tup_dict_warning():
    """Test that using pb_tup_dict emits deprecation warning."""
    with pytest.warns(DeprecationWarning, match="pb_tup_dict"):
        midi_to_table(PALMID, pb_tup_dict={})


def test_df_to_midi_with_ts():
    """Test df_to_midi with time signature parameter."""
    df = midi_to_table(PALMID, notes_only=True)

    _, mid_path = tempfile.mkstemp(suffix=".mid")
    try:
        df_to_midi(df, mid_path, ts="3/4")
        df2 = midi_to_table(mid_path, notes_only=False)
        ts_rows = df2[df2["type"] == "time_signature"]
        assert len(ts_rows) > 0
        ts_other = ts_rows.iloc[0]["other"]
        assert ts_other["numerator"] == 3
        assert ts_other["denominator"] == 4
    finally:
        os.remove(mid_path)


def test_multi_track():
    """Test parsing multi-track MIDI."""
    path = create_test_midi(
        [
            {"pitch": 60, "time": 0, "duration": 480, "track": 0},
            {"pitch": 64, "time": 0, "duration": 480, "track": 1},
            {"pitch": 67, "time": 0, "duration": 480, "track": 2},
        ]
    )
    try:
        df = midi_to_table(path, time_type=float)
        notes = df[df["type"] == "note"]
        assert len(notes) == 3
        assert set(notes["track"].unique()) == {0, 1, 2}
    finally:
        os.remove(path)


if __name__ == "__main__":
    test_basic_note()
    test_consecutive_notes()
    test_overlapping_notes()
    test_midi_to_table()
    test_midi_to_csv()
    test_df_to_midi()
    test_time_type_float()
    test_time_type_fraction()
    test_time_type_int()
    test_notes_only()
    test_display_name()
    test_deprecated_overlapping_notes_warning()
    test_deprecated_pb_tup_dict_warning()
    test_df_to_midi_with_ts()
    test_multi_track()
    print("All tests passed!")
