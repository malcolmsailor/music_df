"""Tests for the optional diff_func on @transform."""

import numpy as np
import pandas as pd
import pytest

from music_df.remove_repeated_bars import (
    _remove_repeated_bars_diff,
    remove_repeated_bars,
)
from music_df.transforms import TRANSFORMS, transform


# ---------------------------------------------------------------------------
# @transform decorator tests
# ---------------------------------------------------------------------------


def test_transform_bare_decorator():
    """@transform (no args) still registers normally and has no diff_func."""

    @transform
    def _test_bare(df):
        return df

    try:
        assert "_test_bare" in TRANSFORMS
        assert not hasattr(_test_bare, "diff_func")
    finally:
        del TRANSFORMS["_test_bare"]


def test_transform_with_diff_func():
    """@transform(diff_func=...) attaches the diff_func attribute."""

    def _my_diff(before, after):
        return set(), set()

    @transform(diff_func=_my_diff)
    def _test_with_diff(df):
        return df

    try:
        assert "_test_with_diff" in TRANSFORMS
        assert _test_with_diff.diff_func is _my_diff
    finally:
        del TRANSFORMS["_test_with_diff"]


def test_transform_duplicate_raises():
    """Registering the same name twice raises ValueError."""

    @transform
    def _test_dup(df):
        return df

    try:
        with pytest.raises(ValueError, match="already registered"):

            @transform
            def _test_dup(df):  # noqa: F811
                return df
    finally:
        del TRANSFORMS["_test_dup"]


# ---------------------------------------------------------------------------
# remove_repeated_bars diff_func tests
# ---------------------------------------------------------------------------


def _make_aac_df() -> pd.DataFrame:
    """Build an AAC dataframe: bars A, A, C where A bars are identical."""
    return pd.DataFrame(
        {
            "type": ["bar", "note", "bar", "note", "bar", "note", "bar"],
            "onset": [0.0, 0.0, 4.0, 4.0, 8.0, 8.0, 12.0],
            "release": [4.0, 1.0, 8.0, 5.0, 12.0, 10.0, 16.0],
            "pitch": [np.nan, 60.0, np.nan, 60.0, np.nan, 64.0, np.nan],
        }
    )


def test_naive_diff_is_wrong_for_remove_repeated_bars():
    """The naive tuple-diff incorrectly flags the shifted C note."""
    df = _make_aac_df()
    before_notes = df[df["type"] == "note"]
    before_tuples = set(
        before_notes[["onset", "release", "pitch"]].itertuples(
            index=False, name=None
        )
    )

    result = remove_repeated_bars(df)
    after_notes = result[result["type"] == "note"]
    after_tuples = set(
        after_notes[["onset", "release", "pitch"]].itertuples(
            index=False, name=None
        )
    )

    naive_removed = before_tuples - after_tuples
    naive_added = after_tuples - before_tuples

    # The C note shifted from onset=8 to onset=4, so naive diff sees both
    # a removal (8.0, 10.0, 64.0) and an addition (4.0, 6.0, 64.0)
    assert len(naive_removed) > 0
    assert len(naive_added) > 0


def test_diff_func_correctly_identifies_removed_bar():
    """diff_func returns only notes from the removed bar, nothing added."""
    df = _make_aac_df()
    result = remove_repeated_bars(df)

    removed, added = _remove_repeated_bars_diff(df, result)

    # Only the note from the second A bar (onset=4, release=5, pitch=60)
    # should be in removed
    assert removed == {(4.0, 5.0, 60.0)}
    assert added == set()


def test_diff_func_no_repeats():
    """When nothing is removed, diff_func returns empty sets."""
    df = pd.DataFrame(
        {
            "type": ["bar", "note", "bar", "note", "bar"],
            "onset": [0.0, 0.0, 4.0, 4.0, 8.0],
            "release": [4.0, 1.0, 8.0, 5.0, 12.0],
            "pitch": [np.nan, 60.0, np.nan, 62.0, np.nan],
        }
    )
    result = remove_repeated_bars(df)
    removed, added = _remove_repeated_bars_diff(df, result)
    assert removed == set()
    assert added == set()


def test_diff_func_registered_on_transform():
    """remove_repeated_bars has diff_func attached via the decorator."""
    assert hasattr(remove_repeated_bars, "diff_func")
    assert remove_repeated_bars.diff_func is _remove_repeated_bars_diff
