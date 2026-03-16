import pandas as pd
import pytest

from music_df.detremolo import detremolo, merge_repeated_notes


def _make_notes_df(notes):
    """Helper: notes is a list of (pitch, onset, release) tuples."""
    return pd.DataFrame(
        [
            {"type": "note", "pitch": p, "onset": o, "release": r}
            for p, o, r in notes
        ]
    )


class TestMergeRepeatedNotes:
    def test_basic_merge(self):
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (60, 0.25, 0.45),
                (60, 0.5, 0.7),
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        notes = result[result["type"] == "note"]
        assert len(notes) == 1
        assert notes.iloc[0]["onset"] == 0.0
        assert notes.iloc[0]["release"] == 0.7

    def test_no_merge_when_gap_too_large(self):
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (60, 0.5, 0.7),
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        notes = result[result["type"] == "note"]
        assert len(notes) == 2

    def test_different_pitches_not_merged(self):
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (64, 0.25, 0.45),
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        notes = result[result["type"] == "note"]
        assert len(notes) == 2

    def test_max_note_duration_none_merges_long_notes(self):
        """With max_note_duration=None, even long notes are merged."""
        df = _make_notes_df(
            [
                (60, 0.0, 2.0),
                (60, 2.05, 4.0),
            ]
        )
        result = merge_repeated_notes(df, max_note_duration=None, max_gap=0.125)
        notes = result[result["type"] == "note"]
        assert len(notes) == 1
        assert notes.iloc[0]["release"] == 4.0

    def test_max_note_duration_set_skips_long_notes(self):
        """With max_note_duration set, long notes are not eligible."""
        df = _make_notes_df(
            [
                (60, 0.0, 2.0),
                (60, 2.05, 4.0),
            ]
        )
        result = merge_repeated_notes(
            df, max_note_duration=0.25, max_gap=0.125
        )
        notes = result[result["type"] == "note"]
        assert len(notes) == 2

    def test_last_note_included_in_merge(self):
        """The last note in a merge group should be included (bug fix)."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (60, 0.25, 0.45),
                (60, 0.5, 0.9),  # this note ends the chain
                (60, 5.0, 5.2),  # separate note, large gap
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        notes = result[result["type"] == "note"]
        assert len(notes) == 2
        assert notes.iloc[0]["release"] == 0.9
        assert notes.iloc[1]["onset"] == 5.0

    def test_single_note_unchanged(self):
        df = _make_notes_df([(60, 0.0, 1.0)])
        result = merge_repeated_notes(df)
        assert len(result) == 1

    def test_non_note_rows_preserved(self):
        df = pd.DataFrame(
            [
                {"type": "note", "pitch": 60, "onset": 0.0, "release": 0.2},
                {"type": "note", "pitch": 60, "onset": 0.25, "release": 0.45},
                {"type": "tempo", "pitch": 0, "onset": 0.0, "release": 0.0},
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        assert len(result[result["type"] == "tempo"]) == 1
        assert len(result[result["type"] == "note"]) == 1

    def test_no_merge_with_intervening_onset(self):
        """Don't merge repeated notes when another note in the same
        instrument sounds in between (fast passage scenario)."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (64, 0.25, 0.45),  # intervening note, different pitch
                (60, 0.3, 0.5),
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        notes = result[result["type"] == "note"]
        pitch_60 = notes[notes["pitch"] == 60]
        assert len(pitch_60) == 2

    def test_merge_without_intervening_onset(self):
        """Merge repeated notes when there are no intervening onsets."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (64, 0.0, 0.15),  # concurrent, not intervening
                (60, 0.25, 0.45),
            ]
        )
        result = merge_repeated_notes(
            df, max_gap=0.125, preserve_outer_voices=False
        )
        notes = result[result["type"] == "note"]
        pitch_60 = notes[notes["pitch"] == 60]
        assert len(pitch_60) == 1
        assert pitch_60.iloc[0]["release"] == 0.45

    def test_merge_at_zero_gap_despite_intervening_onset(self):
        """Contiguous notes (gap=0) always merge, even with intervening
        onsets from the same instrument."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.25),
                (64, 0.1, 0.3),  # overlapping note from same instrument
                (60, 0.25, 0.5),  # gap is exactly 0
            ]
        )
        result = merge_repeated_notes(
            df, max_gap=0.125, preserve_outer_voices=False
        )
        notes = result[result["type"] == "note"]
        pitch_60 = notes[notes["pitch"] == 60]
        assert len(pitch_60) == 1
        assert pitch_60.iloc[0]["release"] == 0.5

    def test_no_merge_when_intervening_onset_at_release_time(self):
        """An intervening note starting exactly when the first note releases
        should block merging."""
        df = _make_notes_df(
            [
                (78, 0.0, 0.125),
                (79, 0.125, 0.25),  # starts exactly at pitch 78's release
                (78, 0.25, 1.0),
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        notes = result[result["type"] == "note"]
        pitch_78 = notes[notes["pitch"] == 78]
        assert len(pitch_78) == 2

    def test_no_instrument_columns_in_df(self):
        """Should work when none of the instrument_columns exist in df."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (60, 0.25, 0.45),
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        notes = result[result["type"] == "note"]
        assert len(notes) == 1


class TestPreserveOuterVoices:
    def test_no_merge_when_bass_status_changes(self):
        """Note C4 is bass alone, then a lower note enters making it no
        longer bass. The two C4 notes should not merge."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.5),  # C4, is bass (alone)
                (48, 0.5, 1.5),  # C3, enters as new bass
                (60, 0.5, 1.0),  # C4, no longer bass
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        pitch_60 = result[result["pitch"] == 60]
        assert len(pitch_60) == 2

    def test_no_merge_when_soprano_status_changes(self):
        """Note C4 is soprano alone, then a higher note enters making it no
        longer soprano. The two C4 notes should not merge."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.5),  # C4, is soprano (alone)
                (72, 0.5, 1.5),  # C5, enters as new soprano
                (60, 0.5, 1.0),  # C4, no longer soprano
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        pitch_60 = result[result["pitch"] == 60]
        assert len(pitch_60) == 2

    def test_merge_when_outer_voice_status_unchanged(self):
        """Two consecutive bass notes should still merge."""
        df = _make_notes_df(
            [
                (48, 0.0, 0.5),  # bass
                (60, 0.0, 1.5),  # soprano throughout
                (48, 0.5, 1.0),  # still bass
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        pitch_48 = result[result["pitch"] == 48]
        assert len(pitch_48) == 1
        assert pitch_48.iloc[0]["release"] == 1.0

    def test_preserve_outer_voices_false_restores_old_behavior(self):
        """With preserve_outer_voices=False, bass status change doesn't
        prevent merging."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.5),  # C4, is bass
                (48, 0.5, 1.5),  # C3, enters as new bass
                (60, 0.5, 1.0),  # C4, no longer bass
            ]
        )
        result = merge_repeated_notes(
            df, max_gap=0.125, preserve_outer_voices=False
        )
        pitch_60 = result[result["pitch"] == 60]
        assert len(pitch_60) == 1
        assert pitch_60.iloc[0]["release"] == 1.0

    def test_solo_note_is_both_bass_and_soprano(self):
        """A note alone is both bass and soprano; two such consecutive
        notes should merge."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.5),
                (60, 0.5, 1.0),
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        assert len(result[result["type"] == "note"]) == 1

    def test_middle_voice_merges_when_outer_voices_constant(self):
        """An inner voice note that stays inner should merge normally."""
        df = _make_notes_df(
            [
                (48, 0.0, 2.0),  # bass throughout
                (72, 0.0, 2.0),  # soprano throughout
                (60, 0.0, 0.5),  # middle voice
                (60, 0.5, 1.0),  # middle voice, same role
            ]
        )
        result = merge_repeated_notes(df, max_gap=0.125)
        pitch_60 = result[result["pitch"] == 60]
        assert len(pitch_60) == 1
        assert pitch_60.iloc[0]["release"] == 1.0


class TestDetremolo:
    def test_wrapper_produces_correct_output(self):
        df = _make_notes_df(
            [
                (60, 0.0, 0.2),
                (60, 0.25, 0.45),
                (60, 0.5, 0.7),
            ]
        )
        result = detremolo(df)
        notes = result[result["type"] == "note"]
        assert len(notes) == 1
        assert notes.iloc[0]["release"] == 0.7

    def test_long_notes_not_merged(self):
        """detremolo defaults to max_tremolo_note_length=0.25."""
        df = _make_notes_df(
            [
                (60, 0.0, 0.5),
                (60, 0.55, 1.0),
            ]
        )
        result = detremolo(df)
        notes = result[result["type"] == "note"]
        assert len(notes) == 2
