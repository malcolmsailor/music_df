import pytest

from music_df import read_krn
from music_df.add_feature import (
    get_bar_relative_onset,
    infer_barlines,
    make_bar_explicit,
    make_instruments_explicit,
    make_time_signatures_explicit,
)
from music_df.read_midi import read_midi
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
