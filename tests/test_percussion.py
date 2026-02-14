"""
Tests for percussion handling in symusic_conv.py and transpose.py.
"""

import os
import tempfile

import pandas as pd
import pytest
import symusic

from music_df.conversions import read_midi_symusic, write_midi_symusic
from music_df.conversions.symusic_conv import symusic_score_to_df
import numpy as np

from music_df.transpose import (
    PERCUSSION_CHANNEL,
    _transpose_pc_string,
    chromatic_transpose,
    transpose_to_key,
)


class TestSymusicDrumTrack:
    """Tests for drum track handling in symusic_conv.py."""

    def test_drum_track_channel(self):
        """Drum tracks should have channel=9, non-drum tracks should have channel=0."""
        score = symusic.Score(480)

        # Create a non-drum track
        melodic_track = symusic.Track()
        melodic_track.notes.append(symusic.Note(time=0, duration=480, pitch=60, velocity=64))
        melodic_track.notes.append(symusic.Note(time=480, duration=480, pitch=62, velocity=64))
        score.tracks.append(melodic_track)

        # Create a drum track
        drum_track = symusic.Track(is_drum=True)
        drum_track.notes.append(symusic.Note(time=0, duration=240, pitch=36, velocity=100))
        drum_track.notes.append(symusic.Note(time=240, duration=240, pitch=38, velocity=100))
        score.tracks.append(drum_track)

        df = symusic_score_to_df(score)
        notes = df[df["type"] == "note"]

        # Track 0 (melodic) should have channel 0
        track0_notes = notes[notes["track"] == 0]
        assert (track0_notes["channel"] == 0).all()

        # Track 1 (drum) should have channel 9
        track1_notes = notes[notes["track"] == 1]
        assert (track1_notes["channel"] == PERCUSSION_CHANNEL).all()

    def test_drum_track_round_trip(self):
        """Drum track is_drum should be preserved through MIDI round-trip."""
        score = symusic.Score(480)

        # Create a drum track
        drum_track = symusic.Track(is_drum=True)
        drum_track.notes.append(symusic.Note(time=0, duration=480, pitch=36, velocity=100))
        score.tracks.append(drum_track)

        # Convert to DataFrame
        df = symusic_score_to_df(score)
        drum_notes = df[(df["type"] == "note") & (df["channel"] == PERCUSSION_CHANNEL)]
        assert len(drum_notes) == 1

        # Write to MIDI and reload
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            temp_path = f.name

        try:
            score.dump_midi(temp_path)
            reloaded_score = symusic.Score(temp_path)

            # The drum track should still be marked as drum
            assert any(track.is_drum for track in reloaded_score.tracks)

            # Convert reloaded score to DataFrame
            reloaded_df = symusic_score_to_df(reloaded_score)
            reloaded_drum_notes = reloaded_df[
                (reloaded_df["type"] == "note")
                & (reloaded_df["channel"] == PERCUSSION_CHANNEL)
            ]
            assert len(reloaded_drum_notes) == 1
        finally:
            os.remove(temp_path)


class TestChromaticTranspose:
    """Tests for percussion handling in chromatic_transpose()."""

    def test_skips_percussion(self):
        """Percussion (channel 9) pitches should not be transposed."""
        df = pd.DataFrame(
            {
                "pitch": [60, 62, 36, 38],
                "channel": [0, 0, PERCUSSION_CHANNEL, PERCUSSION_CHANNEL],
            }
        )

        result = chromatic_transpose(df, interval=5, inplace=False, metadata=False)

        # Melodic pitches (channel 0) should be transposed
        assert result.loc[0, "pitch"] == 65
        assert result.loc[1, "pitch"] == 67

        # Percussion pitches (channel 9) should remain unchanged
        assert result.loc[2, "pitch"] == 36
        assert result.loc[3, "pitch"] == 38

    def test_no_channel_column(self):
        """Without a channel column, all pitches should be transposed."""
        df = pd.DataFrame({"pitch": [60, 62, 64]})

        result = chromatic_transpose(df, interval=3, inplace=False, metadata=False)

        assert result["pitch"].tolist() == [63, 65, 67]


class TestChromaticTransposePcColumns:
    """Tests for pc_columns handling in chromatic_transpose()."""

    def test_numeric_pc_columns(self):
        """Numeric pc_columns should be transposed mod 12."""
        df = pd.DataFrame({"pitch": [60, 62], "pc": [0, 11]})
        df.attrs["pc_columns"] = ("pc",)

        result = chromatic_transpose(df, interval=3, inplace=False, metadata=False)

        assert result["pc"].tolist() == [3, 2]

    def test_string_pc_columns(self):
        """String pc_columns should split/transpose/rejoin correctly."""
        df = pd.DataFrame({"pitch": [60, 62], "key": ["0M", "10.0m"]})
        df.attrs["pc_columns"] = ("key",)

        result = chromatic_transpose(df, interval=5, inplace=False, metadata=False)

        assert result["key"].tolist() == ["5M", "3.0m"]

    def test_string_pc_columns_nan(self):
        """NaN values in string pc_columns should be passed through."""
        df = pd.DataFrame({"pitch": [60, 62], "key": ["0M", float("nan")]})
        df.attrs["pc_columns"] = ("key",)

        result = chromatic_transpose(df, interval=5, inplace=False, metadata=False)

        assert result["key"].iloc[0] == "5M"
        assert pd.isna(result["key"].iloc[1])

    def test_pc_columns_ignore_percussion_mask(self):
        """pc_columns should NOT use the percussion mask."""
        df = pd.DataFrame(
            {
                "pitch": [60, 36],
                "channel": [0, PERCUSSION_CHANNEL],
                "pc": [0, 0],
            }
        )
        df.attrs["pc_columns"] = ("pc",)

        result = chromatic_transpose(df, interval=5, inplace=False, metadata=False)

        # pitch: melodic transposed, percussion unchanged
        assert result.loc[0, "pitch"] == 65
        assert result.loc[1, "pitch"] == 36
        # pc: both rows transposed (percussion mask not applied)
        assert result["pc"].tolist() == [5, 5]


class TestTransposeToKey:
    """Tests for percussion handling in transpose_to_key()."""

    def test_skips_percussion(self):
        """Percussion (channel 9) pitches should not be transposed."""
        df = pd.DataFrame(
            {
                "pitch": [60, 62, 36, 38],
                "channel": [0, 0, PERCUSSION_CHANNEL, PERCUSSION_CHANNEL],
            }
        )
        df.attrs["global_key_sig"] = 0

        result = transpose_to_key(df, new_key_sig=2, inplace=False)

        # Melodic pitches should be transposed (2 sharps = 2 steps along line of fifths)
        # C -> D (60 -> 62), D -> E (62 -> 64)
        assert result.loc[0, "pitch"] == 62
        assert result.loc[1, "pitch"] == 64

        # Percussion pitches should remain unchanged
        assert result.loc[2, "pitch"] == 36
        assert result.loc[3, "pitch"] == 38

    def test_no_channel_column(self):
        """Without a channel column, all pitches should be transposed."""
        df = pd.DataFrame({"pitch": [60, 62, 64]})
        df.attrs["global_key_sig"] = 0

        result = transpose_to_key(df, new_key_sig=2, inplace=False)

        # All pitches should be transposed
        assert result["pitch"].tolist() == [62, 64, 66]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
