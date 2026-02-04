import pandas as pd
import pytest

from music_df import read_krn
from music_df.add_feature import (
    add_bar_durs,
    add_sounding_bass,
    add_time_sig_dur,
    get_bar_relative_onset,
    infer_barlines,
    make_bar_explicit,
    make_instruments_explicit,
    make_time_signatures_explicit,
    split_long_bars,
)
from music_df.read_midi import read_midi
from music_df.sort_df import sort_df
from tests.helpers_for_tests import get_input_kern_paths, get_input_midi_paths


def test_make_time_signatures_explicit(n_kern_files):
    paths = get_input_kern_paths(seed=42)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_krn(path)
        make_time_signatures_explicit(df)


def test_make_bar_explicit(n_kern_files):
    paths = get_input_kern_paths(seed=42)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_krn(path)
        make_bar_explicit(df)


def test_get_bar_relative_onset(n_kern_files):
    paths = get_input_kern_paths(seed=42)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_krn(path)
        get_bar_relative_onset(df)


def test_split_long_bars(n_kern_files):
    # paths = get_input_kern_paths(seed=42)
    paths = [
        # A file with some long measures
        "/Users/malcolm/datasets/humdrum-data/mozart/piano-sonatas/kern/sonata13-3.krn"
    ]
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_krn(path)
        df = make_time_signatures_explicit(df)
        df = sort_df(df)
        before_df = add_bar_durs(add_time_sig_dur(df))
        before_long_bars = before_df["bar_dur"] > before_df["time_sig_dur"]
        assert before_long_bars.any()

        df = split_long_bars(df)

        after_df = add_bar_durs(add_time_sig_dur(df))
        after_long_bars = after_df["bar_dur"] > after_df["time_sig_dur"]
        assert not after_long_bars.any()

        after_df_no_bars = after_df[after_df.type != "bar"].reset_index(drop=True)
        before_df_no_bars = before_df[before_df.type != "bar"].reset_index(drop=True)

        pd.testing.assert_frame_equal(after_df_no_bars, before_df_no_bars)


@pytest.mark.filterwarnings("ignore:note_off event")
def test_infer_barlines(n_kern_files):
    paths = get_input_midi_paths(seed=42)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_midi(path)
        output_df = infer_barlines(df)

        # For the equality comparison to succeed, we need to cast track to float (it
        # has nans in it now)
        df["track"] = df.track.astype(float)
        assert output_df[output_df.type != "bar"].reset_index(drop=True).equals(df)


@pytest.mark.filterwarnings("ignore:note_off event")
def test_make_instruments_explicit(n_kern_files):
    paths = get_input_midi_paths(seed=42)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_midi(path)
        output_df = make_instruments_explicit(df)
        assert output_df.drop("midi_instrument", axis=1).equals(
            df.drop("midi_instrument", axis=1, errors="ignore")
        )
        assert not output_df["midi_instrument"].isna().sum()


class TestAddSoundingBass:
    def test_basic(self):
        """Basic case: three notes, middle one is lowest."""
        df = pd.DataFrame(
            {
                "type": ["note", "note", "note"],
                "pitch": [60.0, 48.0, 55.0],
                "onset": [0.0, 0.0, 1.0],
                "release": [2.0, 1.5, 2.5],
            }
        )
        result = add_sounding_bass(df)
        # At onset 0.0: notes 0,1 sounding, min pitch 48 at idx 1
        # At onset 1.0: notes 0,1,2 sounding (note 1 releases at 1.5 > 1.0), min pitch 48 at idx 1
        assert result["sounding_bass_idx"].tolist() == [1, 1, 1]

    def test_empty_dataframe(self):
        """Empty DataFrame gets NaN column."""
        df = pd.DataFrame({"type": [], "pitch": [], "onset": [], "release": []})
        result = add_sounding_bass(df)
        assert "sounding_bass_idx" in result.columns
        assert len(result) == 0

    def test_no_notes(self):
        """DataFrame with no notes gets all NaN."""
        df = pd.DataFrame(
            {
                "type": ["bar", "time_signature"],
                "pitch": [float("nan"), float("nan")],
                "onset": [0.0, 0.0],
                "release": [4.0, float("nan")],
            }
        )
        result = add_sounding_bass(df)
        assert result["sounding_bass_idx"].isna().all()

    def test_single_note(self):
        """Single note is its own bass."""
        df = pd.DataFrame(
            {
                "type": ["note"],
                "pitch": [60.0],
                "onset": [0.0],
                "release": [1.0],
            }
        )
        result = add_sounding_bass(df)
        assert result["sounding_bass_idx"].tolist() == [0]

    def test_same_pitch_tiebreaker(self):
        """If multiple notes share lowest pitch, first index wins."""
        df = pd.DataFrame(
            {
                "type": ["note", "note", "note"],
                "pitch": [60.0, 60.0, 60.0],
                "onset": [0.0, 0.0, 0.0],
                "release": [1.0, 1.0, 1.0],
            }
        )
        result = add_sounding_bass(df)
        assert result["sounding_bass_idx"].tolist() == [0, 0, 0]

    def test_note_ending_exactly_at_onset(self):
        """A note ending exactly at onset time is not sounding (release > t, not >=)."""
        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [48.0, 60.0],
                "onset": [0.0, 1.0],
                "release": [1.0, 2.0],  # First note ends exactly when second starts
            }
        )
        result = add_sounding_bass(df)
        # At onset 0.0: only note 0 sounding -> bass is 0
        # At onset 1.0: note 0 has release=1.0, so NOT sounding (release > t fails)
        #              only note 1 sounding -> bass is 1
        assert result["sounding_bass_idx"].tolist() == [0, 1]

    def test_non_notes_get_nan(self):
        """Non-note rows get NaN for sounding_bass_idx."""
        df = pd.DataFrame(
            {
                "type": ["bar", "note", "time_signature", "note"],
                "pitch": [float("nan"), 60.0, float("nan"), 48.0],
                "onset": [0.0, 0.0, 0.0, 0.0],
                "release": [4.0, 1.0, float("nan"), 1.0],
            }
        )
        result = add_sounding_bass(df)
        assert pd.isna(result["sounding_bass_idx"].iloc[0])
        assert pd.isna(result["sounding_bass_idx"].iloc[2])
        # Notes should have valid values
        assert result["sounding_bass_idx"].iloc[1] == 3
        assert result["sounding_bass_idx"].iloc[3] == 3

    def test_bass_changes_over_time(self):
        """Bass changes as notes enter and exit."""
        df = pd.DataFrame(
            {
                "type": ["note", "note", "note"],
                "pitch": [60.0, 48.0, 36.0],
                "onset": [0.0, 1.0, 2.0],
                "release": [3.0, 2.0, 3.0],
            }
        )
        result = add_sounding_bass(df)
        # At onset 0.0: only note 0 sounding -> bass is 0
        # At onset 1.0: notes 0,1 sounding, min pitch 48 at idx 1
        # At onset 2.0: notes 0,2 sounding (note 1 ended), min pitch 36 at idx 2
        assert result["sounding_bass_idx"].tolist() == [0, 1, 2]

    def test_preserves_original_index(self):
        """Function works correctly with non-default index."""
        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [60.0, 48.0],
                "onset": [0.0, 0.0],
                "release": [1.0, 1.0],
            },
            index=[10, 20],
        )
        result = add_sounding_bass(df)
        assert result.index.tolist() == [10, 20]
        assert result["sounding_bass_idx"].tolist() == [20, 20]

    def test_percussion_excluded_from_bass(self):
        """Percussion notes are not considered as bass candidates."""
        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [36.0, 60.0],  # 36 = kick drum, lower pitch
                "onset": [0.0, 0.0],
                "release": [1.0, 1.0],
                "channel": [9, 0],  # 9 = percussion
            }
        )
        result = add_sounding_bass(df)
        # Bass should be note 1 (pitch 60), not note 0 (percussion)
        assert result["sounding_bass_idx"].tolist() == [1, 1]

    def test_percussion_gets_bass_assigned(self):
        """Percussion notes still get a sounding_bass_idx value."""
        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [36.0, 60.0],
                "onset": [0.0, 0.0],
                "release": [1.0, 1.0],
                "channel": [9, 0],
            }
        )
        result = add_sounding_bass(df)
        # Percussion note at idx 0 should have bass_idx = 1
        assert result["sounding_bass_idx"].iloc[0] == 1

    def test_no_channel_column_works(self):
        """Function works when channel column is absent."""
        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [48.0, 60.0],
                "onset": [0.0, 0.0],
                "release": [1.0, 1.0],
            }
        )
        result = add_sounding_bass(df)
        assert result["sounding_bass_idx"].tolist() == [0, 0]

    def test_only_percussion_sounding_returns_nan(self):
        """When only percussion is sounding, bass is NaN."""
        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [36.0, 60.0],
                "onset": [0.0, 1.0],  # Percussion alone at onset 0
                "release": [0.5, 2.0],
                "channel": [9, 0],
            }
        )
        result = add_sounding_bass(df)
        assert pd.isna(result["sounding_bass_idx"].iloc[0])  # Percussion-only onset
        assert result["sounding_bass_idx"].iloc[1] == 1  # Normal note
