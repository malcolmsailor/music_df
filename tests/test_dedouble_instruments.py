import numpy as np
import pandas as pd
import pytest

from music_df.dedouble_instruments import (
    CANDIDATE_INSTRUMENT_COLUMNS,
    dedouble_octaves,
    dedouble_octaves_within_instrument,
    dedouble_unisons_across_instruments,
)


def _make_df(rows, columns=("type", "track", "pitch", "onset", "release")):
    return pd.DataFrame(rows, columns=columns)


class TestBasicDoubling:
    def test_two_tracks_identical(self):
        """Two tracks play identical 3-note sequence -> one track removed."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
                ("note", 2, 64, 2.0, 3.0),
            ]
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        assert result.attrs["n_undedoubled_notes"] == 6
        assert result.attrs["n_dedoubled_notes"] == 3
        # Lower-sorted instrument (track 1) is kept
        assert sorted(result["track"].unique()) == [1]


class TestPartialDoubling:
    def test_shared_middle(self):
        """Tracks share notes 2-4 of 6 -> only shared portion removed."""
        df = _make_df(
            [
                # Track 1: 6 notes
                ("note", 1, 58, 0.0, 1.0),
                ("note", 1, 60, 1.0, 2.0),
                ("note", 1, 62, 2.0, 3.0),
                ("note", 1, 64, 3.0, 4.0),
                ("note", 1, 66, 4.0, 5.0),
                ("note", 1, 68, 5.0, 6.0),
                # Track 2: same 3 notes as track 1 positions 1-3
                ("note", 2, 60, 1.0, 2.0),
                ("note", 2, 62, 2.0, 3.0),
                ("note", 2, 64, 3.0, 4.0),
            ]
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        assert result.attrs["n_undedoubled_notes"] == 9
        assert result.attrs["n_dedoubled_notes"] == 6
        # Track 2's 3 doubled notes are removed
        assert (result[result.track == 2].type == "note").sum() == 0


class TestNoDoubling:
    def test_different_content(self):
        """Different content -> nothing removed."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 65, 0.0, 1.0),
                ("note", 2, 67, 1.0, 2.0),
            ]
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 4


class TestMinLength:
    def test_min_length_2_removes(self):
        """2-note shared passage removed with min_length=2."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
            ]
        )
        result = dedouble_unisons_across_instruments(
            df, instrument_columns=["track"], min_length=2
        )
        assert result.attrs["n_dedoubled_notes"] == 2

    def test_min_length_3_keeps(self):
        """2-note shared passage kept with min_length=3."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
            ]
        )
        result = dedouble_unisons_across_instruments(
            df, instrument_columns=["track"], min_length=3
        )
        assert result.attrs["n_dedoubled_notes"] == 4


class TestNonNotePreservation:
    def test_bars_and_time_sigs_survive(self):
        """Non-note rows pass through unchanged."""
        df = pd.DataFrame(
            {
                "type": [
                    "time_signature",
                    "bar",
                    "note",
                    "note",
                    "note",
                    "note",
                    "bar",
                ],
                "track": [np.nan, np.nan, 1, 1, 2, 2, np.nan],
                "pitch": [np.nan, np.nan, 60, 62, 60, 62, np.nan],
                "onset": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 4.0],
                "release": [np.nan, np.nan, 1.0, 2.0, 1.0, 2.0, np.nan],
            }
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        assert (result.type == "time_signature").sum() == 1
        assert (result.type == "bar").sum() == 2


class TestInstrumentColumnAutoDetection:
    def test_detects_track(self):
        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "track": [1, 2],
                "pitch": [60, 65],
                "onset": [0.0, 0.0],
                "release": [1.0, 1.0],
            }
        )
        # Should not raise - auto-detects "track"
        result = dedouble_unisons_across_instruments(df)
        assert result.attrs["dedoubled_instruments"]

    def test_detects_multiple_columns(self):
        df = pd.DataFrame(
            {
                "type": ["note", "note", "note", "note"],
                "track": [1, 1, 2, 2],
                "channel": [0, 1, 0, 1],
                "pitch": [60, 60, 60, 60],
                "onset": [0.0, 0.0, 0.0, 0.0],
                "release": [1.0, 1.0, 1.0, 1.0],
            }
        )
        result = dedouble_unisons_across_instruments(df)
        assert result.attrs["dedoubled_instruments"]

    def test_raises_without_instrument_columns(self):
        df = pd.DataFrame(
            {
                "type": ["note"],
                "pitch": [60],
                "onset": [0.0],
                "release": [1.0],
            }
        )
        with pytest.raises(ValueError, match="No instrument columns found"):
            dedouble_unisons_across_instruments(df)


class TestQuantization:
    def test_slightly_different_onsets_detected_with_quantize(self):
        """Slightly different onsets: detected with quantize=True."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 60, 0.001, 1.001),
                ("note", 2, 62, 1.001, 2.001),
            ]
        )
        result = dedouble_unisons_across_instruments(
            df, instrument_columns=["track"], quantize=True
        )
        assert result.attrs["n_dedoubled_notes"] == 2

    def test_slightly_different_onsets_kept_without_quantize(self):
        """Slightly different onsets: not detected with quantize=False."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 60, 0.001, 1.001),
                ("note", 2, 62, 1.001, 2.001),
            ]
        )
        result = dedouble_unisons_across_instruments(
            df, instrument_columns=["track"], quantize=False
        )
        assert result.attrs["n_dedoubled_notes"] == 4


class TestReleaseTicksPerQuarter:
    def test_sloppy_releases_detected_with_coarse_release_tpq(self):
        """Releases differ by ~0.06 quarters: missed at tpq=16, caught at release_tpq=4."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 60, 0.0, 1.06),
                ("note", 2, 62, 1.0, 2.06),
            ]
        )
        # At tpq=16 for both, 1.0*16=16 vs 1.06*16=17 -> different tokens
        result_fine = dedouble_unisons_across_instruments(
            df, instrument_columns=["track"], ticks_per_quarter=16
        )
        assert result_fine.attrs["n_dedoubled_notes"] == 4

        # With coarse release grid (tpq=4): 1.0*4=4 vs 1.06*4=4 -> same token
        result_coarse = dedouble_unisons_across_instruments(
            df,
            instrument_columns=["track"],
            ticks_per_quarter=16,
            release_ticks_per_quarter=4,
        )
        assert result_coarse.attrs["n_dedoubled_notes"] == 2

    def test_none_falls_back_to_ticks_per_quarter(self):
        """release_ticks_per_quarter=None behaves identically to omitting it."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
            ]
        )
        result_default = dedouble_unisons_across_instruments(
            df, instrument_columns=["track"], ticks_per_quarter=16
        )
        result_none = dedouble_unisons_across_instruments(
            df,
            instrument_columns=["track"],
            ticks_per_quarter=16,
            release_ticks_per_quarter=None,
        )
        pd.testing.assert_frame_equal(result_default, result_none)

    def test_octave_with_coarse_release(self):
        """dedouble_octaves also respects release_ticks_per_quarter."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 72, 0.0, 1.06),
                ("note", 2, 74, 1.0, 2.06),
                ("note", 2, 76, 2.0, 3.06),
            ]
        )
        # Fine release grid -> no match
        result_fine = dedouble_octaves(
            df, instrument_columns=["track"], ticks_per_quarter=16
        )
        assert result_fine.attrs["n_dedoubled_notes"] == 6

        # Coarse release grid -> match
        result_coarse = dedouble_octaves(
            df,
            instrument_columns=["track"],
            ticks_per_quarter=16,
            release_ticks_per_quarter=4,
        )
        assert result_coarse.attrs["n_dedoubled_notes"] == 3


class TestThreeInstruments:
    def test_a_doubles_b_not_c(self):
        """A doubles B but not C -> only B's doubled notes dropped."""
        df = _make_df(
            [
                # Track 1 (A)
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                # Track 2 (B) - same as A
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
                # Track 3 (C) - different
                ("note", 3, 65, 0.0, 1.0),
                ("note", 3, 67, 1.0, 2.0),
            ]
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 4
        # Track 1 kept, track 2 dropped, track 3 kept
        remaining_tracks = sorted(result[result.type == "note"]["track"].unique())
        assert remaining_tracks == [1, 3]


class TestKeepFirstOrdering:
    def test_lower_sorted_instrument_kept(self):
        """Lower-sorted instrument's notes are kept."""
        df = _make_df(
            [
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
            ]
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        remaining_tracks = sorted(result[result.type == "note"]["track"].unique())
        # Track 1 < Track 2 in sort order, so Track 1 kept
        assert remaining_tracks == [1]


class TestExactDoublingUnchanged:
    """Verify dedouble_unisons_across_instruments behavior is identical after refactor."""

    def test_basic_doubling_still_works(self):
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
                ("note", 2, 64, 2.0, 3.0),
            ]
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 3
        assert sorted(result[result.type == "note"]["track"].unique()) == [1]

    def test_no_false_octave_match(self):
        """dedouble_unisons_across_instruments should NOT match notes an octave apart."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 72, 0.0, 1.0),  # octave above
                ("note", 2, 74, 1.0, 2.0),
                ("note", 2, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_unisons_across_instruments(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 6


class TestOctaveDoubling:
    def test_octave_apart_removed(self):
        """Two tracks play same 3-note melody an octave apart -> one removed."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 72, 0.0, 1.0),  # octave above
                ("note", 2, 74, 1.0, 2.0),
                ("note", 2, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_undedoubled_notes"] == 6
        assert result.attrs["n_dedoubled_notes"] == 3


class TestOctaveDoublingKeepsHighVoice:
    def test_high_register_keeps_higher(self):
        """Non-bass octave doubling -> lower voice dropped, higher kept."""
        # Track 3 plays bass below the doubling, so tracks 1/2 are NOT the bass
        df = _make_df(
            [
                ("note", 1, 64, 0.0, 1.0),
                ("note", 1, 66, 1.0, 2.0),
                ("note", 1, 68, 2.0, 3.0),
                ("note", 2, 76, 0.0, 1.0),  # octave above
                ("note", 2, 78, 1.0, 2.0),
                ("note", 2, 80, 2.0, 3.0),
                # Bass instrument below both
                ("note", 3, 36, 0.0, 1.0),
                ("note", 3, 38, 1.0, 2.0),
                ("note", 3, 40, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 6
        # Higher voice (track 2) kept, lower voice (track 1) dropped
        remaining = result[result.type == "note"]
        assert sorted(remaining["track"].unique()) == [2, 3]


class TestOctaveDoublingBassDetection:
    def test_bass_doubling_keeps_lower(self):
        """When lower note IS the global bass, keep lower even in high register."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 72, 0.0, 1.0),
                ("note", 2, 74, 1.0, 2.0),
                ("note", 2, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        remaining = result[result.type == "note"]
        # Lower voice IS the bass → keep lower (track 1)
        assert sorted(remaining["track"].unique()) == [1]

    def test_non_bass_doubling_keeps_higher(self):
        """When another instrument is lower, the doubled passage keeps higher."""
        df = _make_df(
            [
                ("note", 1, 64, 0.0, 1.0),
                ("note", 1, 66, 1.0, 2.0),
                ("note", 1, 68, 2.0, 3.0),
                ("note", 2, 76, 0.0, 1.0),
                ("note", 2, 78, 1.0, 2.0),
                ("note", 2, 80, 2.0, 3.0),
                # Track 3: bass below both
                ("note", 3, 36, 0.0, 1.0),
                ("note", 3, 38, 1.0, 2.0),
                ("note", 3, 40, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 6
        remaining = result[result.type == "note"]
        assert sorted(remaining["track"].unique()) == [2, 3]

    def test_sustained_bass_makes_doubling_non_bass(self):
        """Sustained note from earlier onset makes the doubled note NOT the bass."""
        df = _make_df(
            [
                # Track 3: sustained bass from onset 0 through onset 2
                ("note", 3, 30, 0.0, 3.0),
                # Track 1: octave doubling starting at onset 0
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 72, 0.0, 1.0),
                ("note", 2, 74, 1.0, 2.0),
                ("note", 2, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 4
        remaining = result[result.type == "note"]
        # Track 1 (lower of pair) is NOT bass (track 3 is) → keep higher (track 2)
        assert sorted(remaining["track"].unique()) == [2, 3]


class TestOctaveDoublingKeepsLowVoice:
    def test_low_register_keeps_lower(self):
        """Low-register octave doubling -> higher voice dropped."""
        # Both voices below threshold: mean ~36 and ~48
        df = _make_df(
            [
                ("note", 1, 34, 0.0, 1.0),
                ("note", 1, 36, 1.0, 2.0),
                ("note", 1, 38, 2.0, 3.0),
                ("note", 2, 46, 0.0, 1.0),  # octave above
                ("note", 2, 48, 1.0, 2.0),
                ("note", 2, 50, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 3
        # Lower voice (track 1) should be kept
        remaining = result[result.type == "note"]
        assert sorted(remaining["track"].unique()) == [1]


class TestOctaveDoublingWithChords:
    def test_instrument_playing_chords_with_octave_doubling(self):
        """Cross-instrument octave match detected when one instrument plays chords.

        Track 1 plays two-note chords (e.g., 33+45 at onset 0), with the
        two notes an octave apart. Track 2 doubles the upper note (45).
        In mod12 mode, both chord tones map to the same pitch class, producing
        duplicate tokens in track 1's sequence. The algorithm must collapse
        these duplicates so the suffix array can match track 1 against track 2.
        """
        df = _make_df(
            [
                # Track 1: plays chords (lower + upper, an octave apart)
                ("note", 1, 33, 0.0, 1.0),
                ("note", 1, 45, 0.0, 1.0),
                ("note", 1, 38, 1.0, 2.0),
                ("note", 1, 50, 1.0, 2.0),
                ("note", 1, 45, 2.0, 3.0),
                ("note", 1, 57, 2.0, 3.0),
                # Track 2: doubles the upper note of track 1's chords
                ("note", 2, 45, 0.0, 1.0),
                ("note", 2, 50, 1.0, 2.0),
                ("note", 2, 57, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        # Track 2 is an octave doubling of track 1 and should be removed
        assert result.attrs["n_dedoubled_notes"] == 6


class TestOctaveMinLengthDefault:
    def test_2_note_octave_not_removed_by_default(self):
        """2-note octave doubling NOT removed with default min_length=3."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 72, 0.0, 1.0),
                ("note", 2, 74, 1.0, 2.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 4

    def test_2_note_octave_removed_with_min_length_2(self):
        """2-note octave doubling removed when min_length=2."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 2, 72, 0.0, 1.0),
                ("note", 2, 74, 1.0, 2.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"], min_length=2)
        assert result.attrs["n_dedoubled_notes"] == 2


# ===================================================================
# Within-instrument octave dedoubling tests
# ===================================================================


class TestBasicWithinOctave:
    def test_3_onset_octave_doubling_removed(self):
        """3-onset octave doubling within one instrument -> one voice removed."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 1, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert result.attrs["n_undedoubled_notes"] == 6
        assert result.attrs["n_dedoubled_notes"] == 3

    def test_2_onset_below_min_length_not_removed(self):
        """2-onset octave doubling not removed with default min_length=3."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 4


class TestWithinOctavePitchThreshold:
    def test_non_bass_keeps_higher(self):
        """Non-bass doubling -> lower voice dropped, higher kept."""
        # Track 2 plays bass below, so track 1's lower voice is NOT the bass
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 1, 76, 2.0, 3.0),
                # Bass instrument below
                ("note", 2, 36, 0.0, 1.0),
                ("note", 2, 38, 1.0, 2.0),
                ("note", 2, 40, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 6
        remaining_pitches = sorted(result[result.type == "note"]["pitch"])
        # Higher voice kept from track 1, plus track 2 bass
        assert remaining_pitches == [36, 38, 40, 72, 74, 76]

    def test_low_register_keeps_lower(self):
        """Low-register doubling -> higher voice dropped."""
        # Mean pitch (36+48+38+50+40+52)/6 = 44 < 53 threshold
        df = _make_df(
            [
                ("note", 1, 36, 0.0, 1.0),
                ("note", 1, 48, 0.0, 1.0),
                ("note", 1, 38, 1.0, 2.0),
                ("note", 1, 50, 1.0, 2.0),
                ("note", 1, 40, 2.0, 3.0),
                ("note", 1, 52, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 3
        remaining_pitches = sorted(result[result.type == "note"]["pitch"])
        # Lower voice kept: 36, 38, 40
        assert remaining_pitches == [36, 38, 40]


class TestWithinOctaveMatchReleases:
    def test_different_releases_block_match(self):
        """Different releases block match when match_releases=True."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 0.5),  # different release
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 1.5),  # different release
                ("note", 1, 64, 2.0, 3.0),
                ("note", 1, 76, 2.0, 2.5),  # different release
            ]
        )
        result = dedouble_octaves_within_instrument(
            df, instrument_columns=["track"], match_releases=True
        )
        assert result.attrs["n_dedoubled_notes"] == 6

    def test_different_releases_pass_when_disabled(self):
        """Different releases pass when match_releases=False."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 0.5),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 1.5),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 1, 76, 2.0, 2.5),
            ]
        )
        result = dedouble_octaves_within_instrument(
            df, instrument_columns=["track"], match_releases=False
        )
        assert result.attrs["n_dedoubled_notes"] == 3


class TestWithinOctaveMaxGap:
    def test_gap_breaks_streak_at_max_gap_0(self):
        """Gap at onset 2 breaks streak when max_gap_onsets=0."""
        df = _make_df(
            [
                # Onsets 0,1 have octave pairs; onset 2 does not; onsets 3,4 do
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),  # no octave partner
                ("note", 1, 66, 3.0, 4.0),
                ("note", 1, 78, 3.0, 4.0),
                ("note", 1, 68, 4.0, 5.0),
                ("note", 1, 80, 4.0, 5.0),
            ]
        )
        result = dedouble_octaves_within_instrument(
            df, instrument_columns=["track"], max_gap_onsets=0
        )
        # Neither sub-streak (length 2 each) meets min_length=3
        assert result.attrs["n_dedoubled_notes"] == 9

    def test_gap_continues_streak_at_max_gap_1(self):
        """Gap at onset 2 bridged when max_gap_onsets=1, total streak >= 3."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),  # no octave partner
                ("note", 1, 66, 3.0, 4.0),
                ("note", 1, 78, 3.0, 4.0),
                ("note", 1, 68, 4.0, 5.0),
                ("note", 1, 80, 4.0, 5.0),
            ]
        )
        result = dedouble_octaves_within_instrument(
            df, instrument_columns=["track"], max_gap_onsets=1
        )
        # Streak of 4 matched onsets (0,1,3,4) >= min_length=3 -> 4 notes removed
        assert result.attrs["n_dedoubled_notes"] == 5


class TestWithinOctaveIntervals:
    def test_detects_24_semitone_pairs(self):
        """Default octave_intervals includes 24 (two octaves)."""
        df = _make_df(
            [
                ("note", 1, 48, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),  # 24 apart
                ("note", 1, 50, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 52, 2.0, 3.0),
                ("note", 1, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 3

    def test_custom_intervals_misses_24(self):
        """octave_intervals=(12,) does not detect 24-semitone pairs."""
        df = _make_df(
            [
                ("note", 1, 48, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),  # 24 apart
                ("note", 1, 50, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 52, 2.0, 3.0),
                ("note", 1, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(
            df, instrument_columns=["track"], octave_intervals=(12,)
        )
        assert result.attrs["n_dedoubled_notes"] == 6


class TestWithinOctaveMultipleInstruments:
    def test_each_instrument_dedoubled_independently(self):
        """Two instruments with independent octave doublings."""
        df = _make_df(
            [
                # Track 1: octave doubling
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 1, 76, 2.0, 3.0),
                # Track 2: octave doubling
                ("note", 2, 48, 0.0, 1.0),
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 50, 1.0, 2.0),
                ("note", 2, 62, 1.0, 2.0),
                ("note", 2, 52, 2.0, 3.0),
                ("note", 2, 64, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        # Each track loses 3 notes
        assert result.attrs["n_dedoubled_notes"] == 6
        # Both tracks still present
        remaining_tracks = sorted(result[result.type == "note"]["track"].unique())
        assert remaining_tracks == [1, 2]


class TestWithinOctaveNonNotePreservation:
    def test_bars_preserved(self):
        """Non-note rows (bars, time sigs) pass through unchanged."""
        df = pd.DataFrame(
            {
                "type": ["bar", "note", "note", "note", "note", "note", "note"],
                "track": [np.nan, 1, 1, 1, 1, 1, 1],
                "pitch": [np.nan, 60, 72, 62, 74, 64, 76],
                "onset": [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                "release": [np.nan, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            }
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert (result.type == "bar").sum() == 1
        assert result.attrs["n_dedoubled_notes"] == 3


class TestWithinOctaveNoDoublings:
    def test_no_octave_pairs_nothing_removed(self):
        """No octave pairs at any onset -> nothing removed."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 65, 0.0, 1.0),  # 5 apart, not octave
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 67, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 1, 69, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 6


class TestWithinOctavePitchProximity:
    def test_mixed_register_pairs_tracked_separately(self):
        """Two interval-12 pairs at different registers should form separate
        streaks, each keeping the correct voice for its register.

        Bass pair (39/51, mean 45 < 53): keep lower (39), drop upper (51).
        Treble pair (58/70, mean 64 >= 53): keep upper (70), drop lower (58).
        """
        df = _make_df(
            [
                # onset 0
                ("note", 1, 39, 0.0, 1.0),
                ("note", 1, 51, 0.0, 1.0),  # bass octave pair
                ("note", 1, 58, 0.0, 1.0),
                ("note", 1, 70, 0.0, 1.0),  # treble octave pair
                # onset 1
                ("note", 1, 41, 1.0, 2.0),
                ("note", 1, 53, 1.0, 2.0),
                ("note", 1, 60, 1.0, 2.0),
                ("note", 1, 72, 1.0, 2.0),
                # onset 2
                ("note", 1, 43, 2.0, 3.0),
                ("note", 1, 55, 2.0, 3.0),
                ("note", 1, 62, 2.0, 3.0),
                ("note", 1, 74, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 6
        remaining_pitches = sorted(result[result.type == "note"]["pitch"])
        # Bass: keep lower (39, 41, 43); Treble: keep upper (70, 72, 74)
        assert remaining_pitches == [39, 41, 43, 70, 72, 74]

    def test_same_register_pairs_merge_into_one_streak(self):
        """Two interval-12 pairs at similar registers should merge into one
        streak (only the closest match extends the streak)."""
        df = _make_df(
            [
                # onset 0: one pair
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                # onset 1: two pairs at similar pitches
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 64, 1.0, 2.0),
                ("note", 1, 76, 1.0, 2.0),
                # onset 2: one pair
                ("note", 1, 66, 2.0, 3.0),
                ("note", 1, 78, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        # At least the 3-onset streak (60/72, 62/74, 66/78) should be detected
        assert result.attrs["n_dedoubled_notes"] <= 5


class TestWithinOctaveMixedNotes:
    def test_non_doubled_notes_at_same_onset_survive(self):
        """Non-doubled notes at the same onset are not removed."""
        df = _make_df(
            [
                # Each onset has an octave pair (60/72) plus a non-doubled note
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 67, 0.0, 1.0),  # G4, not doubled
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 69, 1.0, 2.0),  # A4, not doubled
                ("note", 1, 64, 2.0, 3.0),
                ("note", 1, 76, 2.0, 3.0),
                ("note", 1, 71, 2.0, 3.0),  # B4, not doubled
            ]
        )
        result = dedouble_octaves_within_instrument(df, instrument_columns=["track"])
        # 3 doubled notes dropped, 6 remain (3 kept from doubling + 3 non-doubled)
        assert result.attrs["n_dedoubled_notes"] == 6
        remaining_pitches = sorted(result[result.type == "note"]["pitch"])
        # Lower voice IS the bass → keep lower (60,62,64) + non-doubled (67,69,71)
        assert remaining_pitches == [60, 62, 64, 67, 69, 71]


# ===================================================================
# Cross-instrument octave dedoubling with polyphonic instruments
# ===================================================================


class TestPolyphonicCrossInstrumentOctave:
    def test_polyphonic_instrument_octave_doubling(self):
        """Polyphonic instrument doubled at octave by monophonic instrument.

        Track 1 plays 2-note chords; one voice is doubled at the octave
        by track 4's monophonic line. 3 consecutive onsets -> caught with
        min_length=3.
        """
        df = _make_df(
            [
                # Track 1: polyphonic (2-note chords), low register
                ("note", 1, 38, 0.0, 1.0),
                ("note", 1, 45, 0.0, 1.0),
                ("note", 1, 40, 1.0, 2.0),
                ("note", 1, 47, 1.0, 2.0),
                ("note", 1, 42, 2.0, 3.0),
                ("note", 1, 49, 2.0, 3.0),
                # Track 4: monophonic, doubles track 1's upper voice an octave up
                # 45+12=57, 47+12=59, 49+12=61
                ("note", 4, 57, 0.0, 1.0),
                ("note", 4, 59, 1.0, 2.0),
                ("note", 4, 61, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        # The octave doubling should be detected and one voice removed
        assert result.attrs["n_dedoubled_notes"] == 6

    def test_both_instruments_polyphonic(self):
        """Both instruments polyphonic; one voice in each doubles at octave."""
        df = _make_df(
            [
                # Track 1: 2-note chords, low register
                ("note", 1, 36, 0.0, 1.0),
                ("note", 1, 43, 0.0, 1.0),
                ("note", 1, 38, 1.0, 2.0),
                ("note", 1, 45, 1.0, 2.0),
                ("note", 1, 40, 2.0, 3.0),
                ("note", 1, 47, 2.0, 3.0),
                # Track 2: 2-note chords, upper voice doubles track 1 upper at +12
                ("note", 2, 50, 0.0, 1.0),
                ("note", 2, 55, 0.0, 1.0),  # 43+12=55
                ("note", 2, 52, 1.0, 2.0),
                ("note", 2, 57, 1.0, 2.0),  # 45+12=57
                ("note", 2, 54, 2.0, 3.0),
                ("note", 2, 59, 2.0, 3.0),  # 47+12=59
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 9

    def test_no_polyphonic_instruments_skips_second_pass(self):
        """Monophonic instruments only -> suffix array handles everything."""
        df = _make_df(
            [
                ("note", 1, 60, 0.0, 1.0),
                ("note", 1, 62, 1.0, 2.0),
                ("note", 1, 64, 2.0, 3.0),
                ("note", 2, 72, 0.0, 1.0),
                ("note", 2, 74, 1.0, 2.0),
                ("note", 2, 76, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        assert result.attrs["n_dedoubled_notes"] == 3

    def test_mix_suffix_array_and_polyphonic_doublings(self):
        """Some doublings caught by suffix array, others only by polyphonic pass."""
        df = _make_df(
            [
                # Track 1: monophonic, high register
                ("note", 1, 72, 0.0, 1.0),
                ("note", 1, 74, 1.0, 2.0),
                ("note", 1, 76, 2.0, 3.0),
                # Track 2: monophonic, doubles track 1 at octave below
                # suffix array can catch this
                ("note", 2, 60, 0.0, 1.0),
                ("note", 2, 62, 1.0, 2.0),
                ("note", 2, 64, 2.0, 3.0),
                # Track 3: polyphonic, upper voice doubles track 1 at -24
                # Lower voice is NOT at an octave interval from anything
                ("note", 3, 41, 0.0, 1.0),
                ("note", 3, 48, 0.0, 1.0),
                ("note", 3, 43, 1.0, 2.0),
                ("note", 3, 50, 1.0, 2.0),
                ("note", 3, 45, 2.0, 3.0),
                ("note", 3, 52, 2.0, 3.0),
            ]
        )
        result = dedouble_octaves(df, instrument_columns=["track"])
        # Track 1 + Track 2: suffix array drops lower (track 2, 3 notes)
        # Track 3 upper (48,50,52) + Track 1 (72,74,76): polyphonic pass
        #   catches 24-semitone doubling, drops lower (3 notes)
        # Remaining: track 1 (3) + track 3 lower (3) = 6
        assert result.attrs["n_dedoubled_notes"] == 6
