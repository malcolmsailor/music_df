"""
Tests comparing mido-based midi_to_table() with symusic-based read_midi_symusic().
"""

import fractions
import math
import os
import tempfile

import pandas as pd
import pytest

from music_df.conversions import read_midi_symusic, write_midi_symusic
from music_df.midi_parser import midi_to_table

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PALMID = os.path.join(SCRIPT_DIR, "test_files", "misc_Palestrina.mid")


def _compare_note_dfs(
    mido_df: pd.DataFrame,
    symusic_df: pd.DataFrame,
    atol: float = 1e-9,
    compare_tracks: bool = False,
):
    """Compare note DataFrames from mido and symusic.

    Args:
        mido_df: DataFrame from midi_to_table.
        symusic_df: DataFrame from read_midi_symusic.
        atol: Absolute tolerance for float comparisons.
        compare_tracks: If True, compare track numbers. Note that mido and symusic
            may have different track numbering (symusic skips empty tracks).
    """
    # Filter to notes only
    mido_notes = mido_df[mido_df["type"] == "note"].copy()
    symusic_notes = symusic_df[symusic_df["type"] == "note"].copy()

    # Sort both for comparison (don't sort by track since numbering may differ)
    mido_notes = mido_notes.sort_values(
        ["onset", "pitch", "release"]
    ).reset_index(drop=True)
    symusic_notes = symusic_notes.sort_values(
        ["onset", "pitch", "release"]
    ).reset_index(drop=True)

    assert len(mido_notes) == len(symusic_notes), (
        f"Note count mismatch: mido={len(mido_notes)}, symusic={len(symusic_notes)}"
    )

    for i, (_, mido_row), (_, symusic_row) in zip(
        range(len(mido_notes)), mido_notes.iterrows(), symusic_notes.iterrows()
    ):
        # Compare onset
        if isinstance(mido_row["onset"], fractions.Fraction):
            assert mido_row["onset"] == symusic_row["onset"], (
                f"Onset mismatch at row {i}: mido={mido_row['onset']}, symusic={symusic_row['onset']}"
            )
        else:
            assert math.isclose(mido_row["onset"], symusic_row["onset"], abs_tol=atol), (
                f"Onset mismatch at row {i}: mido={mido_row['onset']}, symusic={symusic_row['onset']}"
            )

        # Compare release
        if isinstance(mido_row["release"], fractions.Fraction):
            assert mido_row["release"] == symusic_row["release"], (
                f"Release mismatch at row {i}: mido={mido_row['release']}, symusic={symusic_row['release']}"
            )
        else:
            assert math.isclose(mido_row["release"], symusic_row["release"], abs_tol=atol), (
                f"Release mismatch at row {i}: mido={mido_row['release']}, symusic={symusic_row['release']}"
            )

        # Compare pitch
        assert mido_row["pitch"] == symusic_row["pitch"], (
            f"Pitch mismatch at row {i}: mido={mido_row['pitch']}, symusic={symusic_row['pitch']}"
        )

        # Compare velocity
        assert mido_row["velocity"] == symusic_row["velocity"], (
            f"Velocity mismatch at row {i}: mido={mido_row['velocity']}, symusic={symusic_row['velocity']}"
        )

        # Compare track (optional - symusic may skip empty tracks)
        if compare_tracks:
            assert mido_row["track"] == symusic_row["track"], (
                f"Track mismatch at row {i}: mido={mido_row['track']}, symusic={symusic_row['track']}"
            )


def test_equivalence_float_time():
    """Test that mido and symusic produce equivalent results with float time."""
    mido_df = midi_to_table(PALMID, time_type=float, notes_only=True)
    symusic_df = read_midi_symusic(PALMID, time_type=float, notes_only=True)
    _compare_note_dfs(mido_df, symusic_df)


def test_equivalence_fraction_time():
    """Test that mido and symusic produce equivalent results with Fraction time."""
    mido_df = midi_to_table(PALMID, time_type=fractions.Fraction, notes_only=True)
    symusic_df = read_midi_symusic(PALMID, time_type=fractions.Fraction, notes_only=True)
    _compare_note_dfs(mido_df, symusic_df)


def test_equivalence_int_time():
    """Test that mido and symusic produce equivalent results with int (tick) time."""
    mido_df = midi_to_table(PALMID, time_type=int, notes_only=True)
    symusic_df = read_midi_symusic(PALMID, time_type=int, notes_only=True)
    _compare_note_dfs(mido_df, symusic_df)


def test_equivalence_with_non_note_events():
    """Test that note extraction is equivalent even when non-note events are included."""
    mido_df = midi_to_table(PALMID, time_type=float, notes_only=False)
    symusic_df = read_midi_symusic(PALMID, time_type=float, notes_only=False)
    _compare_note_dfs(mido_df, symusic_df)


def test_round_trip_symusic():
    """Test that reading and writing with symusic preserves notes."""
    original_df = read_midi_symusic(PALMID, time_type=float, notes_only=True)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        temp_path = f.name

    try:
        write_midi_symusic(original_df, temp_path)
        reloaded_df = read_midi_symusic(temp_path, time_type=float, notes_only=True)
        _compare_note_dfs(original_df, reloaded_df)
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
    test_equivalence_float_time()
    test_equivalence_fraction_time()
    test_equivalence_int_time()
    test_equivalence_with_non_note_events()
    test_round_trip_symusic()
    test_display_name()
    test_notes_only_flag()
    test_time_signature_extraction()
    test_tempo_extraction()
    print("All tests passed!")
