"""Tests for the optional diff_func on @transform."""

import numpy as np
import pandas as pd
import pytest

from music_df.quantize_df import _quantize_diff, quantize_df
from music_df.remove_repeated_bars import (
    _find_bars_to_keep,
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

    removed, added, diff_bounds = _remove_repeated_bars_diff(df, result)

    # Only the note from the second A bar (onset=4, release=5, pitch=60)
    # should be in removed
    assert removed == {(4.0, 5.0, 60.0)}
    assert added == set()
    # Span covers both A bars (0-8) in before coords;
    # in after coords the kept bar A spans 0-4
    assert diff_bounds == [(0.0, 8.0, 0.0, 4.0)]


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
    removed, added, diff_bounds = _remove_repeated_bars_diff(df, result)
    assert removed == set()
    assert added == set()
    assert diff_bounds == []


def test_diff_func_registered_on_transform():
    """remove_repeated_bars has diff_func attached via the decorator."""
    assert hasattr(remove_repeated_bars, "diff_func")
    assert remove_repeated_bars.diff_func is _remove_repeated_bars_diff


# ---------------------------------------------------------------------------
# _find_bars_to_keep repeat span tests
# ---------------------------------------------------------------------------


def test_find_bars_to_keep_aa():
    """AA pattern: span covers both bars."""
    fps = [hash("A"), hash("A")]
    kept, spans = _find_bars_to_keep(fps)
    assert kept == [0]
    assert spans == [(0, 0, 1)]


def test_find_bars_to_keep_abab():
    """ABAB pattern: span covers all 4 bars."""
    a, b = hash("A"), hash("B")
    kept, spans = _find_bars_to_keep([a, b, a, b])
    assert kept == [0, 1]
    assert spans == [(0, 1, 3)]


def test_find_bars_to_keep_ababab():
    """ABABAB pattern: span covers all 6 bars."""
    a, b = hash("A"), hash("B")
    kept, spans = _find_bars_to_keep([a, b, a, b, a, b])
    assert kept == [0, 1]
    assert spans == [(0, 1, 5)]


def test_find_bars_to_keep_no_repeats():
    """ABC pattern: no spans."""
    a, b, c = hash("A"), hash("B"), hash("C")
    kept, spans = _find_bars_to_keep([a, b, c])
    assert kept == [0, 1, 2]
    assert spans == []


def test_find_bars_to_keep_aabb():
    """AABB pattern: two iterations produce two spans."""
    a, b = hash("A"), hash("B")
    kept, spans = _find_bars_to_keep([a, a, b, b])
    assert kept == [0, 2]
    assert len(spans) == 2


# ---------------------------------------------------------------------------
# repeat_spans in attrs
# ---------------------------------------------------------------------------


def test_repeat_spans_in_attrs_abab():
    """ABAB: attrs contain a single span covering all 4 bars."""
    df = pd.DataFrame({
        "type": ["bar", "note", "bar", "note", "bar", "note", "bar", "note", "bar"],
        "onset": [0.0, 0.0, 4.0, 4.0, 8.0, 8.0, 12.0, 12.0, 16.0],
        "release": [4.0, 1.0, 8.0, 5.0, 12.0, 9.0, 16.0, 13.0, 20.0],
        "pitch": [np.nan, 60.0, np.nan, 62.0, np.nan, 60.0, np.nan, 62.0, np.nan],
    })
    result = remove_repeated_bars(df)
    # before: 0-16 (full ABAB span); after: 0-8 (kept AB pattern)
    assert result.attrs["repeat_spans"] == [(0.0, 16.0, 0.0, 8.0)]


def test_repeat_spans_empty_when_no_repeats():
    df = pd.DataFrame({
        "type": ["bar", "note", "bar", "note", "bar"],
        "onset": [0.0, 0.0, 4.0, 4.0, 8.0],
        "release": [4.0, 1.0, 8.0, 5.0, 12.0],
        "pitch": [np.nan, 60.0, np.nan, 62.0, np.nan],
    })
    result = remove_repeated_bars(df)
    assert result.attrs["repeat_spans"] == []


# ---------------------------------------------------------------------------
# quantize_df diff_func tests
# ---------------------------------------------------------------------------


def test_quantize_diff_func_registered():
    """quantize_df has diff_func attached via the decorator."""
    assert hasattr(quantize_df, "diff_func")
    assert quantize_df.diff_func is _quantize_diff


def test_quantize_diff_no_drops():
    """Quantization with min_dur doesn't report any removed/added notes."""
    df = pd.DataFrame({
        "type": ["note", "note", "note"],
        "onset": [0.13, 1.01, 2.9],
        "release": [0.87, 2.03, 3.97],
        "pitch": [60, 61, 62],
    })
    result = quantize_df(df, tpq=4)
    removed, added = _quantize_diff(df, result)
    assert removed == set()
    assert added == set()


def test_quantize_diff_with_drops():
    """Quantization with zero_dur_action='remove' correctly reports dropped
    notes without overcounting repositioned notes."""
    df = pd.DataFrame({
        "type": ["note", "note", "note"],
        "onset": [0.0, 0.4, 1.0],
        "release": [0.4, 0.6, 2.0],
        "pitch": [60, 61, 62],
    })
    # At tpq=1, note 0 (onset=0.0, release=0.4) rounds to onset=0, release=0
    # -> zero duration -> dropped
    result = quantize_df(df, tpq=1, zero_dur_action="remove")
    assert len(result[result["type"] == "note"]) == 2

    removed, added = _quantize_diff(df, result)
    assert len(removed) == 1
    assert added == set()
    # The removed tuple should have pitch 60
    assert list(removed)[0][2] == 60


def test_quantize_naive_diff_overcounts():
    """Verify the naive set-diff overcounts for quantize_df, confirming the
    diff_func is needed."""
    df = pd.DataFrame({
        "type": ["note", "note", "note"],
        "onset": [0.13, 1.01, 2.9],
        "release": [0.87, 2.03, 3.97],
        "pitch": [60, 61, 62],
    })
    result = quantize_df(df, tpq=4)

    before_tuples = set(
        df[["onset", "release", "pitch"]].itertuples(index=False, name=None)
    )
    after_tuples = set(
        result[["onset", "release", "pitch"]].itertuples(index=False, name=None)
    )
    naive_removed = before_tuples - after_tuples
    naive_added = after_tuples - before_tuples

    # Naive diff sees repositioned notes as removed+added
    assert len(naive_removed) > 0
    assert len(naive_added) > 0

    # But the diff_func correctly reports nothing
    removed, added = _quantize_diff(df, result)
    assert removed == set()
    assert added == set()


def test_bar_onset_map_aac():
    """AAC: bar_onset_map maps original onsets to shifted onsets."""
    df = _make_aac_df()
    result = remove_repeated_bars(df)
    bar_onset_map = result.attrs["bar_onset_map"]
    # Bar 0 at onset 0 is kept, no shift
    assert bar_onset_map[0.0] == 0.0
    # Bar 2 at onset 8 is kept, shifted left by 4 (one removed bar)
    assert bar_onset_map[8.0] == 4.0
    # Bar 1 at onset 4 was removed, should not be in map
    assert 4.0 not in bar_onset_map
