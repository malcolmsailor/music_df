import io

import pandas as pd

from music_df.harmony.modulation import (
    remove_long_tonicizations,
    remove_short_modulations,
    split_degree_into_primary_and_secondary,
)


def test_no_trailing_slashes_after_chained_calls():
    """Calling remove_long_tonicizations then remove_short_modulations
    should not produce trailing slashes in the degree column.

    Uses pre-split columns to simulate the real pipeline where split columns
    persist across both calls.
    """
    chord_df = pd.read_csv(
        io.StringIO(
            """
onset,primary_degree,secondary_degree,key
0.0,I,I,C
4.0,V,V,C
6.0,V,I,C
7.0,I,I,C
8.0,V,IV,C
9.0,IV,I,C
10.0,I,I,C
"""
        )
    )
    chord_df = remove_long_tonicizations(
        chord_df,
        max_tonicization_num_chords=2,
        min_removal_num_chords=2,
    )
    chord_df = remove_short_modulations(
        chord_df,
        min_modulation_num_chords=3,
        max_removal_num_chords=8,
    )
    trailing = chord_df[chord_df["degree"].str.endswith("/", na=False)]
    assert len(trailing) == 0, (
        f"Trailing slashes found in degree column:\n{trailing[['onset', 'degree']]}"
    )
    double_slash = chord_df[chord_df["degree"].str.contains("//", na=False)]
    assert len(double_slash) == 0, (
        f"Double slashes found in degree column:\n{double_slash[['onset', 'degree']]}"
    )


def test_secondary_degree_not_corrupted_by_remove_long_tonicizations():
    """remove_long_tonicizations should not leave slash-prefixed values in
    secondary_degree when the DataFrame has split columns."""
    chord_df = pd.read_csv(
        io.StringIO(
            """
onset,primary_degree,secondary_degree,key
0.0,I,I,C
1.0,V,V,C
2.0,I,I,C
"""
        )
    )
    result = remove_long_tonicizations(
        chord_df,
        max_tonicization_num_chords=1,
        min_removal_num_chords=1,
    )
    assert not result["secondary_degree"].str.startswith("/").any(), (
        f"Slash-prefixed secondary_degree values:\n{result['secondary_degree']}"
    )
    assert not (result["secondary_degree"] == "").any(), (
        f"Empty secondary_degree values:\n{result[result['secondary_degree'] == '']}"
    )


class TestSecondaryMode:
    """Tests for secondary_mode support in modulation functions."""

    def test_chained_pipeline_with_secondary_mode(self):
        """remove_long_tonicizations then remove_short_modulations should
        preserve and correctly handle secondary_mode."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_mode,key
0.0,I,I,_,C
1.0,V,VI,M,C
2.0,I,VI,M,C
3.0,V,VI,M,C
4.0,I,I,_,C
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df,
            max_tonicization_num_chords=1,
            min_removal_num_chords=1,
        )
        assert "secondary_mode" in result.columns
        assert "degree" in result.columns
        # The tonicization of "VI" with mode "M" should use major key (A, not a)
        tonicized = result[result["key"] != "C"]
        assert len(tonicized) > 0
        assert all(tonicized["key"].str[0].str.isupper())
        # secondary_mode should be "_" where secondary_degree is "I"
        i_mask = result["secondary_degree"] == "I"
        assert (result.loc[i_mask, "secondary_mode"] == "_").all()

        result2 = remove_short_modulations(
            result,
            min_modulation_num_chords=5,
            max_removal_num_chords=10,
        )
        assert "secondary_mode" in result2.columns
        # No trailing slashes or double slashes
        assert not result2["degree"].str.endswith("/", na=False).any()
        assert not result2["degree"].str.contains("//", na=False).any()

    def test_secondary_mode_in_degree_reconstruction(self):
        """When secondary_mode is present, it should appear in the
        reconstructed degree column."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_mode,key
0.0,I,I,_,C
1.0,V,VI,M,C
2.0,V,VI,m,C
3.0,I,I,_,C
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df,
            max_tonicization_num_chords=10,
        )
        assert result.loc[1, "degree"] == "V/VIM"
        assert result.loc[2, "degree"] == "V/VIm"

    def test_secondary_mode_override_in_tonicization(self):
        """secondary_mode_override="M" should force major key
        even when TONICIZATIONS table says minor."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_mode,key
0.0,I,I,_,C
1.0,V,VI,M,C
2.0,IV,VI,M,C
3.0,I,I,_,C
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df,
            max_tonicization_num_chords=1,
        )
        # VI in C major with mode override "M" → A major (not a minor)
        assert result.loc[1, "key"] == "A"
        assert result.loc[2, "key"] == "A"
        assert result.loc[1, "secondary_degree"] == "I"
        assert result.loc[1, "secondary_mode"] == "_"

    def test_no_secondary_mode_is_noop(self):
        """When secondary_mode is absent, behavior is unchanged."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,key
0.0,I,I,C
1.0,V,VI,C
2.0,IV,VI,C
3.0,I,I,C
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df,
            max_tonicization_num_chords=1,
        )
        assert "secondary_mode" not in result.columns
        # Default behavior: VI in C major → a minor
        assert result.loc[1, "key"] == "a"

    def test_replace_spurious_tonics_clears_secondary_mode(self):
        """When a spurious tonic (I/X) is replaced with X,
        secondary_mode should be reset to '_'."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_mode,key
0.0,I,I,_,C
1.0,I,V,M,C
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df,
            max_tonicization_num_chords=10,
        )
        assert result.loc[1, "primary_degree"] == "V"
        assert result.loc[1, "secondary_degree"] == "I"
        assert result.loc[1, "secondary_mode"] == "_"

    def test_split_degree_extracts_mode(self):
        """split_degree_into_primary_and_secondary should extract mode
        suffixes from the secondary degree."""
        df = pd.DataFrame({"degree": ["V/VIM", "V/vim", "V/VI", "I"]})
        result = split_degree_into_primary_and_secondary(df, inplace=False)
        assert result.loc[0, "secondary_mode"] == "M"
        assert result.loc[1, "secondary_mode"] == "m"
        assert result.loc[2, "secondary_mode"] == "_"
        assert result.loc[3, "secondary_mode"] == "_"
        assert result.loc[0, "secondary_degree"] == "VI"
        assert result.loc[1, "secondary_degree"] == "vi"

    def test_remove_short_modulations_sets_secondary_mode(self):
        """When converting a modulation to a tonicization,
        secondary_mode should be set based on the inner key mode."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_mode,key
0.0,I,I,_,C
1.0,V,I,_,G
2.0,I,I,_,C
"""
            )
        )
        result = remove_short_modulations(
            chord_df,
            min_modulation_num_chords=2,
        )
        assert result.loc[1, "secondary_degree"] == "V"
        assert result.loc[1, "secondary_mode"] == "M"
        assert result.loc[1, "degree"] == "V/VM"

    def test_expand_tonicizations_sets_secondary_mode(self):
        """expand_tonicizations should set secondary_mode based on case
        of the degree being moved to secondary position."""
        from music_df.harmony.modulation import expand_tonicizations

        df = pd.DataFrame(
            {
                "primary_degree": ["vi", "V", "vi"],
                "secondary_degree": ["I", "vi", "I"],
                "secondary_mode": ["_", "m", "_"],
                "key": ["C", "C", "C"],
            }
        )
        result = expand_tonicizations(df)
        # Row 0: "vi" moved to secondary → lowercase → mode "m"
        assert result.loc[0, "secondary_mode"] == "m"
        assert result.loc[0, "secondary_degree"] == "vi"
