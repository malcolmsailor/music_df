import math

import pandas as pd
import pytest

from music_df.harmony.matching import percent_chord_df_match, percent_pc_match


class TestPercentPcMatch:
    def test_empty_passage_no_notes(self):
        df = pd.DataFrame({"type": [], "pitch": [], "onset": [], "release": []})
        result = percent_pc_match(df, "047")
        assert math.isnan(result)

    def test_passage_with_only_non_note_types(self):
        df = pd.DataFrame(
            {"type": ["bar", "bar"], "pitch": [None, None], "onset": [0.0, 4.0], "release": [4.0, 8.0]}
        )
        result = percent_pc_match(df, "047")
        assert math.isnan(result)

    def test_all_zero_duration_notes(self):
        df = pd.DataFrame(
            {"type": ["note", "note"], "pitch": [60, 64], "onset": [0.0, 1.0], "release": [0.0, 1.0]}
        )
        result = percent_pc_match(df, "047")
        assert math.isnan(result)

    def test_mixed_zero_nonzero_duration_notes(self):
        df = pd.DataFrame(
            {
                "type": ["note", "note", "note"],
                "pitch": [60, 64, 67],
                "onset": [0.0, 1.0, 2.0],
                "release": [1.0, 1.0, 3.0],  # middle note has zero duration
            }
        )
        # C major chord (0, 4, 7), notes are C (60), E (64), G (67)
        # Only first (dur=1) and third (dur=1) contribute; both match
        result = percent_pc_match(df, "047")
        assert result == 1.0

    def test_normal_case(self):
        df = pd.DataFrame(
            {"type": ["note", "note", "note"], "pitch": [60, 64, 67], "onset": [0.0, 2.0, 3.0], "release": [2.0, 3.0, 4.0]}
        )
        result = percent_pc_match(df, "047")
        assert result == 1.0

    def test_partial_match(self):
        df = pd.DataFrame(
            {"type": ["note", "note", "note"], "pitch": [60, 64, 67], "onset": [0.0, 2.0, 3.0], "release": [2.0, 3.0, 4.0]}
        )
        # A major chord (9, 1, 4) - only E (64 -> pc 4) matches
        result = percent_pc_match(df, "914")
        assert result == 0.25  # E has duration 1 out of total 4


class TestPercentChordDfMatch:
    def test_empty_chord_df(self):
        music_df = pd.DataFrame(
            {"type": ["note"], "pitch": [60], "onset": [0.0], "release": [1.0]}
        )
        chord_df = pd.DataFrame({"onset": [], "release": [], "chord_pcs": []})
        result = percent_chord_df_match(music_df, chord_df)
        assert math.isnan(result["macroaverage"])
        assert math.isnan(result["microaverage"])

    def test_chord_with_no_notes(self):
        music_df = pd.DataFrame(
            {"type": ["note"], "pitch": [60], "onset": [0.0], "release": [1.0]}
        )
        # Chord spans 2.0-3.0, but note is at 0.0-1.0
        chord_df = pd.DataFrame({"onset": [2.0], "release": [3.0], "chord_pcs": ["047"]})
        result = percent_chord_df_match(music_df, chord_df)
        assert math.isnan(result["macroaverage"])

    def test_mixed_valid_invalid_chords(self):
        music_df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [60, 64],
                "onset": [0.0, 0.0],
                "release": [1.0, 1.0],
            }
        )
        chord_df = pd.DataFrame(
            {
                "onset": [0.0, 2.0],
                "release": [1.0, 3.0],
                "chord_pcs": ["047", "047"],  # first chord has notes, second doesn't
            }
        )
        result = percent_chord_df_match(music_df, chord_df)
        # First chord: C and E match C major (0, 4, 7) -> 100%
        # Second chord: no notes -> NaN (excluded from macroaverage)
        assert result["macroaverage"] == 1.0

    def test_all_chords_have_no_notes(self):
        music_df = pd.DataFrame(
            {"type": ["note"], "pitch": [60], "onset": [0.0], "release": [1.0]}
        )
        chord_df = pd.DataFrame(
            {"onset": [2.0, 4.0], "release": [3.0, 5.0], "chord_pcs": ["047", "047"]}
        )
        result = percent_chord_df_match(music_df, chord_df)
        assert math.isnan(result["macroaverage"])

    def test_zero_duration_chord(self):
        music_df = pd.DataFrame(
            {"type": ["note"], "pitch": [60], "onset": [0.0], "release": [1.0]}
        )
        # Zero-duration chord at onset=1.0, release=1.0
        chord_df = pd.DataFrame({"onset": [1.0], "release": [1.0], "chord_pcs": ["047"]})
        result = percent_chord_df_match(music_df, chord_df)
        # No notes fall within [1.0, 1.0] since note releases at 1.0
        assert math.isnan(result["macroaverage"])

    def test_normal_case(self):
        music_df = pd.DataFrame(
            {
                "type": ["note", "note", "note"],
                "pitch": [60, 64, 67],
                "onset": [0.0, 0.0, 0.0],
                "release": [1.0, 1.0, 1.0],
            }
        )
        chord_df = pd.DataFrame({"onset": [0.0], "release": [1.0], "chord_pcs": ["047"]})
        result = percent_chord_df_match(music_df, chord_df)
        assert result["macroaverage"] == 1.0
        assert result["microaverage"] == 1.0

    def test_index_alignment_with_non_note_rows(self):
        """Verify that non-note rows don't receive chord match values after slicing.

        This tests a bug where slice_df reindexes the DataFrame, causing assignments
        to go to wrong rows when using the sliced DataFrame's indices.
        """
        music_df = pd.DataFrame(
            {
                "type": ["bar", "time_signature", "note", "note", "bar", "note", "note"],
                "pitch": [None, None, 60, 64, None, 67, 71],
                "onset": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                "release": [2.0, 2.0, 1.0, 2.0, 4.0, 3.0, 4.0],
            }
        )
        chord_df = pd.DataFrame(
            {
                "onset": [0.0, 2.0],
                "release": [2.0, 4.0],
                "chord_pcs": ["047", "72B"],
            }
        )
        result = percent_chord_df_match(music_df, chord_df)
        result_df = result["music_df"]

        # Non-note rows should have NaN for percent_chord_match
        non_note_mask = result_df["type"] != "note"
        assert result_df.loc[non_note_mask, "percent_chord_match"].isna().all()

        # Non-note rows should have empty string for chord_pcs
        assert (result_df.loc[non_note_mask, "chord_pcs"] == "").all()

        # Note rows should have values assigned
        note_mask = result_df["type"] == "note"
        assert not result_df.loc[note_mask, "percent_chord_match"].isna().any()

        # Verify chord_pcs alignment: notes at onset 0-2 get "047", notes at 2-4 get "72B"
        first_chord_notes = (result_df["type"] == "note") & (result_df["onset"] < 2.0)
        second_chord_notes = (result_df["type"] == "note") & (result_df["onset"] >= 2.0)
        assert (result_df.loc[first_chord_notes, "chord_pcs"] == "047").all()
        assert (result_df.loc[second_chord_notes, "chord_pcs"] == "72B").all()
