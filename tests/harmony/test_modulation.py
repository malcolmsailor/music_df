import io

import pandas as pd

from music_df.harmony.modulation import (
    remove_long_tonicizations,
    remove_short_modulations,
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
