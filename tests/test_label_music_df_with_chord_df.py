"""Tests for label_music_df_with_chord_df unmatched parameter."""

import io

import pandas as pd
import pytest

from music_df.chord_df import label_music_df_with_chord_df


@pytest.fixture
def music_df_with_early_notes():
    """Music DF where notes start at onset 0 but chords start at onset 2."""
    return pd.read_csv(
        io.StringIO(
            """\
type,pitch,onset,release
bar,,0.0,4.0
note,60,0.0,1.0
note,64,1.0,2.0
note,62,2.0,3.0
note,67,3.0,4.0
bar,,4.0,8.0
note,66,4.0,6.0
note,67,6.0,8.0"""
        )
    )


@pytest.fixture
def chord_df_starting_at_2():
    """Chord DF where first annotation is at onset 2, not 0."""
    return pd.DataFrame(
        {
            "onset": [2.0, 5.0, 7.0],
            "key": ["C", "G", "G"],
            "degree": ["I", "V", "I"],
            "quality": ["M", "M", "M"],
            "inversion": [0.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def chord_df_starting_at_0():
    """Chord DF where first annotation is at onset 0."""
    return pd.DataFrame(
        {
            "onset": [0.0, 3.0, 5.0, 7.0],
            "key": ["C", "C", "G", "G"],
            "degree": ["I", "V", "V", "I"],
            "quality": ["M", "M", "M", "M"],
            "inversion": [0.0, 1.0, 0.0, 0.0],
        }
    )


class TestBackfill:
    def test_notes_before_first_chord_get_first_chord_values(
        self, music_df_with_early_notes, chord_df_starting_at_2
    ):
        result = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_2
        )
        # Notes at onset 0.0 and 1.0 are before first chord at 2.0
        early_notes = result[(result["type"] == "note") & (result["onset"] < 2.0)]
        assert len(early_notes) == 2
        for _, row in early_notes.iterrows():
            assert row["key"] == "C"
            assert row["degree"] == "I"
            assert row["quality"] == "M"
            assert row["inversion"] == 0.0

    def test_notes_after_first_chord_unchanged(
        self, music_df_with_early_notes, chord_df_starting_at_2
    ):
        result = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_2
        )
        # Note at onset 3.0 should still get chord at 2.0 (backward merge)
        note_at_3 = result[(result["type"] == "note") & (result["onset"] == 3.0)]
        assert note_at_3.iloc[0]["key"] == "C"
        assert note_at_3.iloc[0]["degree"] == "I"

    def test_no_unmatched_notes_no_change(
        self, music_df_with_early_notes, chord_df_starting_at_0
    ):
        result = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_0
        )
        # All notes should have chord values (no NaN in key for notes)
        notes = result[result["type"] == "note"]
        assert not notes["key"].isna().any()

    def test_nonnote_rows_still_get_null_token(
        self, music_df_with_early_notes, chord_df_starting_at_2
    ):
        result = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_2
        )
        bars = result[result["type"] == "bar"]
        assert (bars["key"] == "na").all()
        assert (bars["degree"] == "na").all()
        assert (bars["quality"] == "na").all()
        assert bars["inversion"].isna().all()


class TestDrop:
    def test_notes_before_first_chord_are_dropped(
        self, music_df_with_early_notes, chord_df_starting_at_2
    ):
        result = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_2, unmatched="drop"
        )
        notes = result[result["type"] == "note"]
        # Notes at onset 0.0 and 1.0 should be gone
        assert 0.0 not in notes["onset"].values
        assert 1.0 not in notes["onset"].values
        # Notes at onset 2.0+ should remain
        assert 2.0 in notes["onset"].values
        assert 3.0 in notes["onset"].values

    def test_index_is_reset_after_drop(
        self, music_df_with_early_notes, chord_df_starting_at_2
    ):
        result = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_2, unmatched="drop"
        )
        assert list(result.index) == list(range(len(result)))

    def test_nonnote_rows_before_first_chord_kept(
        self, music_df_with_early_notes, chord_df_starting_at_2
    ):
        """Bars before first chord should not be dropped even with drop mode."""
        result = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_2, unmatched="drop"
        )
        bars = result[result["type"] == "bar"]
        # Bar at onset 0.0 should still be present
        assert 0.0 in bars["onset"].values
        assert (bars["key"] == "na").all()

    def test_no_unmatched_notes_no_change(
        self, music_df_with_early_notes, chord_df_starting_at_0
    ):
        result_drop = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_0, unmatched="drop"
        )
        result_backfill = label_music_df_with_chord_df(
            music_df_with_early_notes, chord_df_starting_at_0, unmatched="backfill"
        )
        pd.testing.assert_frame_equal(result_drop, result_backfill)


class TestDtypeMismatch:
    def test_int_note_onset_float_chord_onset(self):
        """merge_asof requires matching dtypes; int vs float should not crash."""
        music_df = pd.DataFrame(
            {
                "type": ["bar", "note", "note"],
                "pitch": [float("nan"), 60.0, 64.0],
                "onset": [0, 0, 2],  # int64
                "release": [4, 2, 4],
            }
        )
        chord_df = pd.DataFrame(
            {
                "onset": [0.0, 2.0],  # float64
                "key": ["C", "G"],
            }
        )
        assert music_df["onset"].dtype != chord_df["onset"].dtype
        result = label_music_df_with_chord_df(music_df, chord_df, columns_to_add=("key",))
        notes = result[result["type"] == "note"]
        assert notes.iloc[0]["key"] == "C"
        assert notes.iloc[1]["key"] == "G"
