"""Tests for attrs-based skip logic in sort_df and quantize_df."""

import numpy as np
import pandas as pd
import pytest

from music_df.quantize_df import quantize_df
from music_df.sort_df import sort_df


def _make_unsorted_df():
    return pd.DataFrame(
        {
            "type": ["bar", "note", "note", "note"],
            "pitch": [np.nan, 67, 60, 64],
            "onset": [0.0, 2.0, 0.0, 1.0],
            "release": [4.0, 3.0, 1.0, 2.0],
        }
    )


# ── sort_df ──────────────────────────────────────────────────────────────


class TestSortDfSkip:
    def test_sort_sets_attr(self):
        df = _make_unsorted_df()
        result = sort_df(df)
        assert result.attrs["sorted"] is True

    def test_skip_on_already_sorted(self):
        df = _make_unsorted_df()
        sorted_df = sort_df(df)
        # Second call should skip (return a copy without re-sorting)
        result = sort_df(sorted_df)
        assert result.attrs["sorted"] is True
        pd.testing.assert_frame_equal(result, sorted_df)

    def test_skip_returns_copy_when_not_inplace(self):
        df = sort_df(_make_unsorted_df())
        result = sort_df(df)
        assert result is not df

    def test_skip_returns_same_when_inplace(self):
        df = sort_df(_make_unsorted_df())
        result = sort_df(df, inplace=True)
        assert result is df

    def test_force_re_sorts_despite_attr(self):
        df = sort_df(_make_unsorted_df())
        # Manually break the sort order but leave the attr
        df.iloc[0], df.iloc[-1] = df.iloc[-1].copy(), df.iloc[0].copy()
        df = df.reset_index(drop=True)
        result = sort_df(df, force=True)
        assert result.attrs["sorted"] is True
        # Should actually be sorted now
        notes = result[result.type == "note"]
        assert (notes["onset"].diff().dropna() >= -1e-9).all()


# ── quantize_df ──────────────────────────────────────────────────────────


def _make_unquantized_df():
    return pd.DataFrame(
        {
            "type": ["note", "note", "note"],
            "pitch": [60, 61, 62],
            "onset": [0.01, 1.03, 1.95],
            "release": [0.99, 2.01, 3.05],
        }
    )


class TestQuantizeDfSkip:
    def test_quantize_sets_attrs(self):
        df = _make_unquantized_df()
        result = quantize_df(df, tpq=4)
        assert result.attrs["quantized_tpq"] == 4
        assert result.attrs["quantized_ticks_out"] is False
        assert result.attrs["quantized_zero_dur_action"] == "min_dur"

    def test_skip_on_same_params(self):
        df = _make_unquantized_df()
        q1 = quantize_df(df, tpq=4)
        q2 = quantize_df(q1, tpq=4)
        # Should be the same object (skip returns df directly)
        assert q2 is q1

    def test_no_skip_on_different_tpq(self):
        df = _make_unquantized_df()
        q1 = quantize_df(df, tpq=4)
        q2 = quantize_df(q1, tpq=16)
        assert q2 is not q1
        assert q2.attrs["quantized_tpq"] == 16

    def test_no_skip_on_different_zero_dur_action(self):
        df = _make_unquantized_df()
        q1 = quantize_df(df, tpq=4, zero_dur_action="min_dur")
        q2 = quantize_df(q1, tpq=4, zero_dur_action="preserve")
        assert q2 is not q1

    def test_preserves_sorted_attr_and_re_sorts(self):
        df = _make_unquantized_df()
        sorted_df = sort_df(df)
        assert sorted_df.attrs["sorted"] is True
        result = quantize_df(sorted_df, tpq=4)
        assert result.attrs["sorted"] is True
        # Verify actually sorted
        assert (result["onset"].diff().dropna() >= -1e-9).all()

    def test_preserves_custom_attrs(self):
        df = _make_unquantized_df()
        df.attrs["_test_marker"] = True
        result = quantize_df(df, tpq=4)
        assert result.attrs.get("_test_marker") is True


# ── transforms with force=True ───────────────────────────────────────────


class TestForceSort:
    def test_merge_notes_sorts_despite_sorted_attr(self):
        from music_df.merge_notes import merge_notes

        df = pd.DataFrame(
            {
                "type": ["note", "note", "note"],
                "pitch": [60, 60, 64],
                "onset": [0.0, 0.5, 2.0],
                "release": [1.0, 1.5, 3.0],
            }
        )
        df.attrs["sorted"] = True
        result = merge_notes(df)
        assert result.attrs["sorted"] is True
        assert (result["onset"].diff().dropna() >= -1e-9).all()

    def test_slice_df_sorts_despite_sorted_attr(self):
        from music_df.slice_df import slice_df

        df = pd.DataFrame(
            {
                "type": ["note", "note"],
                "pitch": [60, 64],
                "onset": [0.0, 2.0],
                "release": [3.0, 4.0],
            }
        )
        df.attrs["sorted"] = True
        result = slice_df(df, slice_boundaries=[1.0])
        assert result.attrs["sorted"] is True
        assert (result["onset"].diff().dropna() >= -1e-9).all()
