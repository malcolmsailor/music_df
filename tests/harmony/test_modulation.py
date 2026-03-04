import io

import pandas as pd

from music_df.harmony.modulation import (
    _reconstruct_degree_column,
    remove_long_tonicizations,
    remove_phantom_keys,
    remove_short_modulations,
    replace_spurious_tonics,
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

    def test_nested_secondary_across_modulation(self):
        """When removing a short modulation, nested secondary degrees should
        be resolved in the inner key then expressed relative to the outer key.

        #VII/VIm in Eb is Bo7 (B diminished 7th targeting C minor).
        In F minor, C is the 5th degree, so this should become #VII/V, not
        #VII/#V.
        """
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,release,chord_pcs,primary_degree,secondary_degree,secondary_mode,inversion,key,quality
45.0,46.0,47a1,#VII,I,_,0.0,f,o7
46.0,47.0,58b2,#VII,VI,m,2.0,Eb,o7
47.0,48.0,047,V,I,_,0.0,f,M
"""
            ),
        )
        result = remove_short_modulations(
            chord_df, min_modulation_num_chords=2
        )
        assert result.loc[1, "secondary_degree"] == "V"
        assert result.loc[1, "primary_degree"] == "#VII"
        assert result.loc[1, "key"] == "f"
        assert result.loc[1, "degree"] == "#VII/VM"

    def test_expand_tonicizations_sets_secondary_mode(self):
        """expand_tonicizations should set secondary_mode based on quality
        of the degree being moved to secondary position."""
        from music_df.harmony.modulation import expand_tonicizations

        df = pd.DataFrame(
            {
                "primary_degree": ["VI", "V", "VI"],
                "secondary_degree": ["I", "VI", "I"],
                "secondary_mode": ["_", "m", "_"],
                "quality": ["m", "M", "m"],
                "key": ["C", "C", "C"],
            }
        )
        result = expand_tonicizations(df, quality_col="quality")
        # Row 0: quality "m" → secondary_mode "m"
        assert result.loc[0, "secondary_mode"] == "m"
        assert result.loc[0, "secondary_degree"] == "VI"


class TestSecondaryAlteration:
    """Tests for secondary_alteration support in remove_long_tonicizations."""

    def test_secondary_alteration_incorporated_in_key(self):
        """secondary_alteration='b' with secondary_degree='III' in key F
        should tonicize to Ab, not A."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_alteration,key
0.0,I,I,_,F
1.0,I,III,b,F
2.0,V,III,b,F
3.0,I,I,_,F
"""
            ),
            dtype={"secondary_alteration": str},
        )
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=1
        )
        assert result.loc[1, "key"] == "Ab"
        assert result.loc[2, "key"] == "Ab"
        assert result.loc[1, "secondary_degree"] == "I"
        assert result.loc[2, "secondary_degree"] == "I"
        assert result.loc[1, "secondary_alteration"] == "-"
        assert result.loc[2, "secondary_alteration"] == "-"

    def test_sharp_secondary_alteration(self):
        """secondary_alteration='#' with secondary_degree='IV' in key F
        should tonicize to B, not Bb."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_alteration,secondary_mode,key
0.0,I,I,_,_,F
1.0,I,IV,#,M,F
2.0,V,IV,#,M,F
3.0,I,I,_,_,F
"""
            ),
            dtype={"secondary_alteration": str},
        )
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=1
        )
        assert result.loc[1, "key"] == "B"
        assert result.loc[2, "key"] == "B"
        assert result.loc[1, "secondary_alteration"] == "-"

    def test_different_alterations_not_collapsed(self):
        """Adjacent rows with same secondary_degree but different
        secondary_alteration should not be collapsed together."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_alteration,key
0.0,V,III,b,F
1.0,IV,III,b,F
2.0,V,III,#,F
3.0,IV,III,#,F
4.0,I,I,_,F
"""
            ),
            dtype={"secondary_alteration": str},
        )
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=1
        )
        # bIII and #III should be treated as different tonicizations
        # bIII in F → Ab; #III in F → Bb (III=A, #=A#→Bb)
        assert result.loc[0, "key"] == "Ab"
        assert result.loc[1, "key"] == "Ab"
        assert result.loc[2, "key"] == "Bb"
        assert result.loc[3, "key"] == "Bb"

class TestRemovePhantomKeys:
    """Tests for remove_phantom_keys."""

    def test_absorbed_into_following_key(self):
        """C → D(V/IV) → G: IV of D = G, dist(C,G)=1, dist(G,G)=0 → all to G."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,C
1.0,V/IV,D
2.0,I,G
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert result.loc[1, "key"] == "G"
        assert result.loc[1, "degree"] == "V"
        assert list(result["key"]) == ["C", "G", "G"]

    def test_absorbed_into_preceding_key(self):
        """C → G(V/IV, ii/IV) → A: IV of G = C, dist(C,C)=0, dist(A,C)=3 → all to C."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,C
1.0,V/IV,G
2.0,II/IV,G
3.0,I,A
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert list(result["key"]) == ["C", "C", "C", "A"]
        assert result.loc[1, "degree"] == "V"
        assert result.loc[2, "degree"] == "II"

    def test_different_degree_and_spurious_tonic(self):
        """C → D(V/VI, I/VI) → A: VI of D = b, dist(A,b)=1 < dist(C,b)=2.

        After absorbing into A: V/ii and I/ii → replace_spurious_tonics → II.
        """
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,C
1.0,V/VI,D
2.0,I/VI,D
3.0,I,A
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert list(result["key"]) == ["C", "A", "A", "A"]
        assert result.loc[1, "degree"] == "V/II"
        # I/II → spurious tonic → II (uppercased: primary degrees are uppercase)
        assert result.loc[2, "degree"] == "II"

    def test_at_beginning(self):
        """D(V/IV, ii/IV) → G: only following neighbor G. IV of D = G → becomes I."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,V/IV,D
1.0,II/IV,D
2.0,I,G
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert list(result["key"]) == ["G", "G", "G"]
        assert result.loc[0, "degree"] == "V"
        assert result.loc[1, "degree"] == "II"

    def test_at_end(self):
        """C → D(V/VI, II/VI): only preceding neighbor C. VI of D = b → VII in C."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,C
1.0,V/VI,D
2.0,II/VI,D
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert list(result["key"]) == ["C", "C", "C"]
        assert result.loc[1, "degree"] == "V/VII"
        assert result.loc[2, "degree"] == "II/VII"

    def test_split_needed(self):
        """Ab → Bb(V/IV, ii/IV, V/V) → C: IV=Eb→Ab (dist 1), V=F→C (dist 1)."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,Ab
1.0,V/IV,Bb
2.0,II/IV,Bb
3.0,V/V,Bb
4.0,I,C
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert list(result["key"]) == ["Ab", "Ab", "Ab", "C", "C"]
        # Eb expressed in Ab = V
        assert result.loc[1, "degree"] == "V/V"
        assert result.loc[2, "degree"] == "II/V"
        # F expressed in C = IV
        assert result.loc[3, "degree"] == "V/IV"

    def test_not_all_tonicization_unchanged(self):
        """C → G(V/V, IV) → D: G has non-tonicized IV → not phantom, unchanged."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,C
1.0,V/V,G
2.0,IV,G
3.0,I,D
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert list(result["key"]) == ["C", "G", "G", "D"]
        assert result.loc[1, "degree"] == "V/V"
        assert result.loc[2, "degree"] == "IV"

    def test_mode_mismatch(self):
        """C → D(V/V) → G: V of D = A major. In G, A = II but needs "M" suffix.

        The default mode for II in G would be minor (a), but A major requires
        an explicit mode override via secondary_mode.
        """
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,secondary_mode,key
0.0,I,I,_,C
1.0,V,V,_,D
2.0,I,I,_,G
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert result.loc[1, "key"] == "G"
        assert result.loc[1, "secondary_degree"] == "II"
        assert result.loc[1, "secondary_mode"] == "M"
        assert result.loc[1, "degree"] == "V/IIM"

    def test_modular_distance(self):
        """B → Gb(V/V, ii/V) → C: V of Gb = Db, dist(B,Db)=2, dist(C,Db)=5.

        Db is closer to B than C on the circle of fifths (going the short way
        around, using mod-12 arithmetic). Both chords should be assigned to B.
        """
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,B
1.0,V/V,Gb
2.0,II/V,Gb
3.0,I,C
"""
            )
        )
        result = remove_phantom_keys(chord_df)
        assert list(result["key"]) == ["B", "B", "B", "C"]
        # Db in B major gives bbIII (Db is a diminished 3rd above B)
        assert result.loc[1, "degree"] == "V/bbIII"
        assert result.loc[2, "degree"] == "II/bbIII"


class TestReconstructDegreeColumn:
    """Tests for bugs in _reconstruct_degree_column."""

    def test_primary_alteration_preserved(self):
        """primary_alteration should be prepended to primary_degree."""
        df = pd.DataFrame(
            {
                "primary_degree": ["II", "VII", "VI", "IV"],
                "primary_alteration": ["b", "#", "b", "#"],
                "secondary_degree": ["I", "I", "I", "I"],
            }
        )
        result = _reconstruct_degree_column(df)
        assert result.tolist() == ["bII", "#VII", "bVI", "#IV"]

    def test_primary_alteration_sentinel_ignored(self):
        """primary_alteration '_' (null sentinel) should not appear in output."""
        df = pd.DataFrame(
            {
                "primary_degree": ["V", "I"],
                "primary_alteration": ["_", "_"],
                "secondary_degree": ["I", "I"],
            }
        )
        result = _reconstruct_degree_column(df)
        assert result.tolist() == ["V", "I"]

    def test_secondary_alteration_preserved(self):
        """secondary_alteration should be prepended to secondary_degree."""
        df = pd.DataFrame(
            {
                "primary_degree": ["V", "IV"],
                "primary_alteration": ["_", "_"],
                "secondary_degree": ["II", "VII"],
                "secondary_alteration": ["b", "#"],
            }
        )
        result = _reconstruct_degree_column(df)
        assert result.tolist() == ["V/bII", "IV/#VII"]

    def test_redundant_mode_suffix_stripped_major_key(self):
        """'/IM' in a major key is redundant (tonic forced major = already major)
        and should be stripped. '/Im' in a major key is meaningful (parallel minor)
        and should be kept."""
        df = pd.DataFrame(
            {
                "primary_degree": ["I", "V", "V"],
                "secondary_degree": ["I", "I", "I"],
                "secondary_mode": ["M", "m", "_"],
                "key": ["C", "C", "C"],
            }
        )
        result = _reconstruct_degree_column(df)
        assert result.tolist() == ["I", "V/Im", "V"]

    def test_redundant_mode_suffix_stripped_minor_key(self):
        """'/Im' in a minor key is redundant (tonic forced minor = already minor)
        and should be stripped. '/IM' in a minor key is meaningful (parallel major)
        and should be kept."""
        df = pd.DataFrame(
            {
                "primary_degree": ["I", "V", "V"],
                "secondary_degree": ["I", "I", "I"],
                "secondary_mode": ["m", "M", "_"],
                "key": ["c", "c", "c"],
            }
        )
        result = _reconstruct_degree_column(df)
        assert result.tolist() == ["I", "V/IM", "V"]

    def test_all_three_bugs_together(self):
        """Reproducer from the bug report exercising all three issues at once."""
        df = pd.DataFrame(
            {
                "primary_degree": ["II", "VII", "V", "I"],
                "primary_alteration": ["b", "#", "_", "_"],
                "secondary_degree": ["I", "I", "II", "I"],
                "secondary_alteration": ["_", "_", "b", "_"],
                "secondary_mode": ["_", "_", "M", "M"],
                "key": ["C", "C", "C", "C"],
            }
        )
        result = _reconstruct_degree_column(df)
        assert result.tolist() == ["bII", "#VII", "V/bIIM", "I"]


class TestJoinedFormatWithAlterations:
    """Tests for joined-format input (degree column) containing altered secondary degrees.

    These regression tests verify that the canonical split (single_degree_to_split_degrees)
    correctly separates alterations and that downstream functions handle them properly.
    """

    def test_remove_long_tonicizations_with_flat_secondary(self):
        """V/bIII in Db should tonicize to Fb (= E)."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,key
0.0,I,Db
1.0,V/bIII,Db
2.0,IV/bIII,Db
3.0,I,Db
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=1
        )
        assert result.loc[1, "key"] == "E"
        assert result.loc[2, "key"] == "E"
        assert result.loc[1, "degree"] == "V"
        assert result.loc[2, "degree"] == "IV"

    def test_tonicization_census_distinguishes_alterations(self):
        """V/bVI and V/#VI should count as separate tonicizations."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,release,degree,key
0.0,1.0,I,C
1.0,2.0,V/bVI,C
2.0,3.0,IV/bVI,C
3.0,4.0,V/#VI,C
4.0,5.0,IV/#VI,C
5.0,6.0,I,C
"""
            )
        )
        from music_df.harmony.modulation import tonicization_census

        result = tonicization_census(chord_df)
        assert len(result) == 2
        # Check they are counted separately
        assert result.iloc[0]["n_chords"] == 2
        assert result.iloc[1]["n_chords"] == 2


class TestReplaceSpuriousTonicsAlterations:
    """Tests for alteration/case bugs in replace_spurious_tonics."""

    def test_secondary_alteration_moved_to_primary(self):
        """When I/bV becomes V, secondary_alteration 'b' should move to
        primary_alteration."""
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "primary_degree": ["I", "I"],
                "primary_alteration": ["_", "_"],
                "secondary_degree": ["I", "V"],
                "secondary_alteration": ["_", "b"],
                "key": ["C", "C"],
            }
        )
        result = replace_spurious_tonics(df)
        assert result.loc[1, "primary_degree"] == "V"
        assert result.loc[1, "primary_alteration"] == "b"
        assert result.loc[1, "secondary_alteration"] == "_"

    def test_spurious_tonic_replacement(self):
        """When I/V is replaced, primary_degree should become V."""
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "primary_degree": ["I", "I"],
                "secondary_degree": ["I", "V"],
                "key": ["C", "C"],
            }
        )
        result = replace_spurious_tonics(df)
        assert result.loc[1, "primary_degree"] == "V"


class TestInversionAwareCounting:
    """Tests for inversion-aware chord counting in remove_long_tonicizations
    and remove_short_modulations."""

    def test_inversions_count_as_one_in_tonicization(self):
        """V/V with three different inversions = 1 distinct chord.
        Old raw count was 3 (> 2 → removed). New distinct count is 1
        (<= 2 → kept)."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,inversion,key
0.0,I,0,C
1.0,V/V,0,C
2.0,V/V,1,C
3.0,V/V,2,C
4.0,I,0,C
"""
            )
        )
        # 1 distinct chord <= 2 → kept as tonicization
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=2
        )
        assert result.loc[1, "degree"] == "V/V"
        assert result.loc[1, "key"] == "C"

        # 1 distinct chord > 0 → removed
        result2 = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=0
        )
        assert result2.loc[1, "key"] == "G"
        assert result2.loc[2, "key"] == "G"
        assert result2.loc[3, "key"] == "G"

    def test_distinct_chords_not_removed_under_threshold(self):
        """V/V and IV/V are 2 distinct chords (inversions don't add),
        so max_tonicization_num_chords=2 should keep the tonicization."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,inversion,key
0.0,I,0,C
1.0,V/V,0,C
2.0,V/V,1,C
3.0,IV/V,0,C
4.0,I,0,C
"""
            )
        )
        # 2 distinct chords <= 2 → kept
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=2
        )
        assert result.loc[1, "degree"] == "V/V"

        # 2 distinct chords > 1 → removed
        result2 = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=1
        )
        assert result2.loc[1, "key"] == "G"

    def test_quality_aware_derepeat(self):
        """V/V (M) followed by V7/V (Mm7) should NOT be collapsed during
        de-repeat because they have different qualities."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,primary_degree,secondary_degree,quality,key
0.0,I,I,M,C
1.0,V,V,M,C
2.0,V,V,Mm7,C
3.0,I,I,M,C
"""
            )
        )
        result = remove_long_tonicizations(
            chord_df, max_tonicization_num_chords=1
        )
        # 2 distinct chords (M vs Mm7) > 1 → should be removed
        assert result.loc[1, "key"] == "G"
        assert result.loc[2, "key"] == "G"

    def test_inversions_count_as_one_in_modulation(self):
        """Different inversions of I in a modulated key should count as 1
        distinct chord for min_modulation_num_chords."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,inversion,key
0.0,I,0,C
1.0,I,0,G
2.0,I,1,G
3.0,I,2,G
4.0,I,0,C
"""
            )
        )
        # 1 distinct chord < 2 → removed
        result = remove_short_modulations(
            chord_df, min_modulation_num_chords=2
        )
        assert result.loc[1, "key"] == "C"
        assert result.loc[2, "key"] == "C"
        assert result.loc[3, "key"] == "C"

    def test_modulation_kept_with_enough_distinct_chords(self):
        """I and V are 2 distinct chords, so min_modulation_num_chords=2
        should keep the modulation."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,inversion,key
0.0,I,0,C
1.0,I,0,G
2.0,V,0,G
3.0,I,0,C
"""
            )
        )
        # 2 distinct chords >= 2 → kept
        result = remove_short_modulations(
            chord_df, min_modulation_num_chords=2
        )
        assert result.loc[1, "key"] == "G"
        assert result.loc[2, "key"] == "G"

    def test_max_removal_num_chords_inversion_aware(self):
        """max_removal_num_chords in remove_short_modulations should use
        inversion-aware counting."""
        chord_df = pd.read_csv(
            io.StringIO(
                """
onset,degree,inversion,key
0.0,I,0,C
1.0,I,0,G
2.0,I,1,G
3.0,I,2,G
4.0,I,0,C
"""
            )
        )
        # duration=3.0 < min_modulation_duration=4.0, and
        # 1 distinct chord <= max_removal_num_chords=1 → removal allowed
        result = remove_short_modulations(
            chord_df,
            min_modulation_duration=4.0,
            max_removal_num_chords=1,
        )
        assert result.loc[1, "key"] == "C"
