import io

import pandas as pd
import pytest

from music_df.chord_df import compare_chords, compare_keys


# ── compare_chords ───────────────────────────────────────────────────────────


class TestCompareChords:
    """Tests for compare_chords."""

    @pytest.fixture()
    def joined_df(self):
        return pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0, 3.0],
                "key": ["C", "C", "G", "G"],
                "degree": ["I", "V", "IV", "I"],
                "quality": ["M", "M", "M", "M"],
                "inversion": [0, 0, 1, 0],
            }
        )

    def test_identical_returns_zero(self, joined_df):
        assert compare_chords(joined_df, joined_df) == 0

    def test_single_degree_change(self, joined_df):
        other = joined_df.copy()
        other.loc[2, "degree"] = "V"
        assert compare_chords(joined_df, other) == 1

    def test_single_key_change(self, joined_df):
        other = joined_df.copy()
        other.loc[0, "key"] = "G"
        assert compare_chords(joined_df, other) == 1

    def test_multiple_column_diff_still_one_row(self, joined_df):
        """Changing key *and* degree on the same row counts as 1 difference."""
        other = joined_df.copy()
        other.loc[1, "key"] = "F"
        other.loc[1, "degree"] = "ii"
        assert compare_chords(joined_df, other) == 1

    def test_all_rows_different(self, joined_df):
        other = joined_df.copy()
        other["key"] = "F"
        assert compare_chords(joined_df, other) == len(joined_df)

    def test_explicit_cols(self, joined_df):
        other = joined_df.copy()
        other.loc[0, "key"] = "F"
        other.loc[0, "degree"] = "ii"
        # Only comparing degree → 1 diff
        assert compare_chords(joined_df, other, cols=["degree"]) == 1
        # Only comparing key → 1 diff
        assert compare_chords(joined_df, other, cols=["key"]) == 1
        # Comparing both → still 1 diff (same row)
        assert compare_chords(joined_df, other, cols=["key", "degree"]) == 1

    def test_split_format_auto_detected(self):
        df1 = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "C"],
                "primary_degree": ["I", "V"],
                "primary_alteration": ["_", "_"],
                "secondary_degree": ["I", "I"],
                "secondary_alteration": ["_", "_"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            }
        )
        df2 = df1.copy()
        df2.loc[1, "primary_degree"] = "IV"
        assert compare_chords(df1, df2) == 1

    def test_split_format_with_secondary_mode(self):
        df1 = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "C"],
                "primary_degree": ["V", "V"],
                "primary_alteration": ["_", "_"],
                "secondary_degree": ["V", "V"],
                "secondary_alteration": ["_", "_"],
                "secondary_mode": ["M", "M"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            }
        )
        df2 = df1.copy()
        df2.loc[0, "secondary_mode"] = "m"
        assert compare_chords(df1, df2) == 1

    def test_length_mismatch_raises(self, joined_df):
        with pytest.raises(AssertionError, match="same length"):
            compare_chords(joined_df, joined_df.iloc[:2])

    def test_ignores_index_values(self, joined_df):
        """Non-default index values shouldn't cause false mismatches."""
        other = joined_df.copy()
        other.index = [10, 20, 30, 40]
        assert compare_chords(joined_df, other) == 0


# ── compare_keys ─────────────────────────────────────────────────────────────


class TestCompareKeys:
    """Tests for compare_keys."""

    def test_identical(self):
        df = pd.DataFrame({"key": ["C", "C", "G", "G"]})
        result = compare_keys(df, df)
        assert result == {"n_rows": 0, "n_regions": 0}

    def test_all_different(self):
        df1 = pd.DataFrame({"key": ["C", "C", "C"]})
        df2 = pd.DataFrame({"key": ["G", "G", "G"]})
        result = compare_keys(df1, df2)
        assert result == {"n_rows": 3, "n_regions": 1}

    def test_partial_overlap(self):
        df1 = pd.DataFrame({"key": ["C", "C", "C", "G"]})
        df2 = pd.DataFrame({"key": ["C", "C", "G", "G"]})
        result = compare_keys(df1, df2)
        # Row 2: df1=C, df2=G → differs. Rows 0,1,3 match.
        assert result["n_rows"] == 1
        # Regions: [0,1] both C, [2] C vs G, [3] G vs G → 1 differs
        assert result["n_regions"] == 1

    def test_ffill_empty_strings(self):
        df1 = pd.DataFrame({"key": ["C", "", "", "G", ""]})
        df2 = pd.DataFrame({"key": ["C", "", "F", "", ""]})
        result = compare_keys(df1, df2)
        # After ffill: df1=[C,C,C,G,G], df2=[C,C,F,F,F]
        # Rows 2,3,4 differ → n_rows=3
        assert result["n_rows"] == 3
        # Unified boundaries at 0, 2(df2 changes), 3(df1 changes)
        # Region [0,1]: C vs C → same
        # Region [2]: C vs F → diff
        # Region [3,4]: G vs F → diff
        assert result["n_regions"] == 2

    def test_ffill_nan(self):
        df1 = pd.DataFrame({"key": ["C", float("nan"), "G"]})
        df2 = pd.DataFrame({"key": ["C", float("nan"), "C"]})
        result = compare_keys(df1, df2)
        # After ffill: df1=[C,C,G], df2=[C,C,C]
        assert result["n_rows"] == 1
        assert result["n_regions"] == 1

    def test_length_mismatch_raises(self):
        df1 = pd.DataFrame({"key": ["C", "C"]})
        df2 = pd.DataFrame({"key": ["C"]})
        with pytest.raises(AssertionError, match="same length"):
            compare_keys(df1, df2)

    def test_many_alternating_regions(self):
        df1 = pd.DataFrame({"key": ["C", "G", "C", "G"]})
        df2 = pd.DataFrame({"key": ["G", "C", "G", "C"]})
        result = compare_keys(df1, df2)
        assert result["n_rows"] == 4
        assert result["n_regions"] == 4

    def test_with_remove_long_tonicizations(self):
        """Realistic before/after using remove_long_tonicizations."""
        from music_df.harmony.modulation import remove_long_tonicizations

        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,key
0.0,I,I,C
1.0,V,V,C
2.0,I,V,C
3.0,IV,V,C
4.0,I,I,C
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=1
        )
        cmp = compare_keys(chord_df, result)
        # Tonicization rows should now have a different key
        assert cmp["n_rows"] > 0
        assert cmp["n_regions"] > 0
