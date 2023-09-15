import pandas as pd
import pytest

from music_df import read_krn
from music_df.add_feature import (
    add_bar_durs,
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
