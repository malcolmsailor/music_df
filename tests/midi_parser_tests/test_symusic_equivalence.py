"""
Tests for symusic-based MIDI parsing functions.

This module tests read_midi_symusic() and write_midi_symusic() from
music_df.conversions.symusic_conv.
"""

import fractions
import math
import os
import tempfile

import pandas as pd
import pytest

from music_df.conversions import read_midi_symusic, write_midi_symusic

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PALMID = os.path.join(SCRIPT_DIR, "test_files", "misc_Palestrina.mid")


def test_read_midi_symusic_float_time():
    """Test read_midi_symusic with float time."""
    df = read_midi_symusic(PALMID, time_type=float, notes_only=True)
    assert len(df) > 0
    assert all(isinstance(t, float) for t in df["onset"])


def test_read_midi_symusic_fraction_time():
    """Test read_midi_symusic with Fraction time."""
    df = read_midi_symusic(PALMID, time_type=fractions.Fraction, notes_only=True)
    assert len(df) > 0
    assert all(isinstance(t, fractions.Fraction) for t in df["onset"])


def test_read_midi_symusic_int_time():
    """Test read_midi_symusic with int (tick) time."""
    df = read_midi_symusic(PALMID, time_type=int, notes_only=True)
    assert len(df) > 0
    assert all(isinstance(t, int) for t in df["onset"])


def test_round_trip_symusic():
    """Test that reading and writing with symusic preserves notes."""
    original_df = read_midi_symusic(PALMID, time_type=float, notes_only=True)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        temp_path = f.name

    try:
        write_midi_symusic(original_df, temp_path)
        reloaded_df = read_midi_symusic(temp_path, time_type=float, notes_only=True)

        # Sort both for comparison
        original_sorted = original_df.sort_values(
            ["onset", "pitch", "release"]
        ).reset_index(drop=True)
        reloaded_sorted = reloaded_df.sort_values(
            ["onset", "pitch", "release"]
        ).reset_index(drop=True)

        assert len(original_sorted) == len(reloaded_sorted)

        for i in range(len(original_sorted)):
            orig = original_sorted.iloc[i]
            reload = reloaded_sorted.iloc[i]
            assert math.isclose(orig["onset"], reload["onset"], abs_tol=1e-9)
            assert math.isclose(orig["release"], reload["release"], abs_tol=1e-9)
            assert orig["pitch"] == reload["pitch"]
            assert orig["velocity"] == reload["velocity"]
    finally:
        os.remove(temp_path)


def test_display_name():
    """Test that display_name parameter works correctly."""
    custom_name = "custom_filename.mid"
    df = read_midi_symusic(PALMID, display_name=custom_name)
    assert (df["filename"] == custom_name).all()


def test_notes_only_flag():
    """Test that notes_only flag excludes non-note events."""
    notes_only_df = read_midi_symusic(PALMID, notes_only=True)
    all_events_df = read_midi_symusic(PALMID, notes_only=False)

    assert set(notes_only_df["type"].unique()) == {"note"}
    assert len(all_events_df) >= len(notes_only_df)


def test_time_signature_extraction():
    """Test that time signatures are extracted correctly."""
    df = read_midi_symusic(PALMID, notes_only=False)
    time_sigs = df[df["type"] == "time_signature"]

    for _, ts in time_sigs.iterrows():
        assert "numerator" in ts["other"]
        assert "denominator" in ts["other"]
        assert isinstance(ts["other"]["numerator"], int)
        assert isinstance(ts["other"]["denominator"], int)


def test_tempo_extraction():
    """Test that tempo events are extracted correctly."""
    df = read_midi_symusic(PALMID, notes_only=False)
    tempos = df[df["type"] == "tempo"]

    for _, tempo in tempos.iterrows():
        assert "tempo" in tempo["other"]
        assert isinstance(tempo["other"]["tempo"], (int, float))


if __name__ == "__main__":
    test_read_midi_symusic_float_time()
    test_read_midi_symusic_fraction_time()
    test_read_midi_symusic_int_time()
    test_round_trip_symusic()
    test_display_name()
    test_notes_only_flag()
    test_time_signature_extraction()
    test_tempo_extraction()
    print("All tests passed!")
