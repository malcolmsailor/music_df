from music_df.segment_df import (
    get_eligible_onsets,
    get_eligible_releases,
)

from music_df import read_krn
from tests.helpers_for_tests import get_input_kern_paths


def test_get_eligible_onsets(n_kern_files):
    paths = get_input_kern_paths(seed=42)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_krn(path)
        indices = get_eligible_onsets(df, keep_onsets_together=True)
        for idx in indices:
            onset = df.loc[idx, "onset"]
            assert df[df.onset == onset].index.min() == idx


def test_get_eligible_releases(n_kern_files):
    paths = get_input_kern_paths(seed=42)
    for i, path in enumerate(paths):
        print(f"{i}/{len(paths)}: {path}")
        df = read_krn(path)
        release_df = get_eligible_releases(df, keep_releases_together=True)
        assert (
            release_df.sort_values(inplace=False, ignore_index=True).values
            == release_df.values
        ).all()
        for idx, val in release_df.items():
            pitch = df.loc[idx, "pitch"]
            assert df[df.release == val].pitch.max() == pitch
