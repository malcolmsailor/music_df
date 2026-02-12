"""Round-trip tests for chord_df format conversions.

Tests that converting between joined, split, and rn formats preserves
information correctly. See notes/chord_df_formats.md for format details.
"""

import pandas as pd
import pytest

from music_df.chord_df import (
    DEFAULT_NULL_CHORD_TOKEN,
    get_quality_for_merging,
    inversion_number_to_figure,
    single_degree_to_split_degrees,
    split_degrees_to_single_degree,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_joined_df(degree: str, **kwargs) -> pd.DataFrame:
    """Build a single-row joined-format DataFrame."""
    return pd.DataFrame({"degree": [degree], **kwargs})


def _make_split_df(
    primary_degree: str,
    primary_alteration: str,
    secondary_degree: str,
    secondary_alteration: str,
    secondary_mode: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    data = {
        "primary_degree": [primary_degree],
        "primary_alteration": [primary_alteration],
        "secondary_degree": [secondary_degree],
        "secondary_alteration": [secondary_alteration],
        **kwargs,
    }
    if secondary_mode is not None:
        data["secondary_mode"] = [secondary_mode]
    return pd.DataFrame(data)


# ── Class 1: joined → split → joined ────────────────────────────────────


class TestJoinedToSplitRoundTrip:
    @pytest.mark.parametrize(
        "degree",
        [
            "I",
            "IV",
            "VII",
            "bVI",
            "#IV",
            "bbVII",
            "##I",
            "V/V",
            "VII/V",
            "V/bVII",
            "#VI/bII",
            "V/Vm",
            "V/VM",
            "VII/bIIm",
            "III/VM",
        ],
    )
    def test_round_trip(self, degree):
        df = _make_joined_df(degree)
        split_df = single_degree_to_split_degrees(df, inplace=False)
        result_df = split_degrees_to_single_degree(split_df, inplace=False)
        assert result_df["degree"].iloc[0] == degree

    def test_null_chord_round_trip_requires_type_column(self):
        """split_degrees_to_single_degree only maps null tokens back to
        DEFAULT_NULL_CHORD_TOKEN when a 'type' column is present."""
        df = pd.DataFrame(
            {"type": ["bar"], "degree": [DEFAULT_NULL_CHORD_TOKEN]}
        )
        split_df = single_degree_to_split_degrees(df, inplace=False)
        result_df = split_degrees_to_single_degree(split_df, inplace=False)
        assert result_df["degree"].iloc[0] == DEFAULT_NULL_CHORD_TOKEN


# ── Class 2: split → joined → split ─────────────────────────────────────


class TestSplitToJoinedRoundTrip:
    @pytest.mark.parametrize(
        "primary_degree, primary_alteration, secondary_degree, secondary_alteration, secondary_mode",
        [
            ("I", "_", "I", "_", "_"),
            ("V", "_", "I", "_", "_"),
            ("IV", "_", "I", "_", "_"),
            ("VII", "_", "I", "_", "_"),
            ("VI", "b", "I", "_", "_"),
            ("IV", "#", "I", "_", "_"),
            ("V", "_", "V", "_", "_"),
            ("VII", "_", "V", "_", "_"),
            ("V", "_", "VII", "b", "_"),
            ("VI", "#", "II", "b", "_"),
            ("V", "_", "V", "_", "m"),
            ("V", "_", "V", "_", "M"),
            ("VII", "#", "V", "_", "m"),
            ("VII", "_", "II", "b", "m"),
            ("III", "_", "V", "_", "M"),
            # secondary_mode on secondary_degree="I": /Im and /IM survive the
            # /I$ regex strip because the suffix isn't bare /I
            ("I", "_", "I", "_", "m"),
            ("I", "_", "I", "_", "M"),
            ("V", "_", "I", "_", "m"),
        ],
    )
    def test_round_trip(
        self,
        primary_degree,
        primary_alteration,
        secondary_degree,
        secondary_alteration,
        secondary_mode,
    ):
        df = _make_split_df(
            primary_degree,
            primary_alteration,
            secondary_degree,
            secondary_alteration,
            secondary_mode,
        )
        joined_df = split_degrees_to_single_degree(df, inplace=False)
        result_df = single_degree_to_split_degrees(joined_df, inplace=False)

        assert result_df["primary_degree"].iloc[0] == primary_degree
        assert result_df["primary_alteration"].iloc[0] == primary_alteration
        assert result_df["secondary_degree"].iloc[0] == secondary_degree
        assert result_df["secondary_alteration"].iloc[0] == secondary_alteration
        assert result_df["secondary_mode"].iloc[0] == secondary_mode

    def test_null_chord_round_trip_with_type_column(self):
        """Null chords round-trip when a 'type' column is present.

        A non-null row with "/" must also be present so that str.split
        produces 2 columns — otherwise single_degree_to_split_degrees
        fills secondary columns with defaults rather than null_chord_token.
        """
        na = DEFAULT_NULL_CHORD_TOKEN
        df = pd.DataFrame(
            {
                "type": ["bar", "note"],
                "primary_degree": [na, "V"],
                "primary_alteration": [na, "_"],
                "secondary_degree": [na, "V"],
                "secondary_alteration": [na, "_"],
                "secondary_mode": [na, "_"],
            }
        )
        joined = split_degrees_to_single_degree(df, inplace=False)
        assert joined.loc[0, "degree"] == na
        result = single_degree_to_split_degrees(joined, inplace=False)
        assert result["primary_degree"].iloc[0] == na
        assert result["primary_alteration"].iloc[0] == na
        assert result["secondary_degree"].iloc[0] == na
        assert result["secondary_alteration"].iloc[0] == na
        assert result["secondary_mode"].iloc[0] == na


# ── Class 3: split → rn output ──────────────────────────────────────────


class TestSplitToRnOutput:
    """One-way correctness checks: split + quality + inversion → rn string.

    Follows the merge_annotations pipeline: get_quality_for_merging strips
    "7" from quality, and inversion_number_to_figure converts int to figure.
    No rn → split parser exists yet (see notes/chord_df_formats.md Future Work).
    """

    @pytest.mark.parametrize(
        "primary_degree, primary_alteration, secondary_degree, secondary_alteration, "
        "secondary_mode, raw_quality, inversion, expected_rn",
        [
            ("I", "_", "I", "_", "_", "M", 0, "IM"),
            ("V", "_", "I", "_", "_", "M", 1, "VM6"),
            ("I", "_", "I", "_", "_", "m", 0, "Im"),
            ("I", "_", "I", "_", "_", "m7", 0, "Im7"),
            ("I", "_", "I", "_", "_", "Mm7", 3, "IMm42"),
            ("VII", "#", "V", "_", "_", "d7", 2, "#VIId43/V"),
            ("V", "_", "V", "_", "m", "M", 0, "VM/Vm"),
            ("V", "_", "VII", "b", "_", "M", 1, "VM6/bVII"),
            ("VI", "b", "I", "_", "_", "M", 0, "bVIM"),
            ("V", "_", "I", "_", "m", "M", 0, "VM/Im"),
        ],
    )
    def test_split_to_rn(
        self,
        primary_degree,
        primary_alteration,
        secondary_degree,
        secondary_alteration,
        secondary_mode,
        raw_quality,
        inversion,
        expected_rn,
    ):
        figure = inversion_number_to_figure(inversion, raw_quality)
        display_quality = get_quality_for_merging(raw_quality)
        df = pd.DataFrame(
            {
                "primary_degree": [primary_degree],
                "primary_alteration": [primary_alteration],
                "secondary_degree": [secondary_degree],
                "secondary_alteration": [secondary_alteration],
                "secondary_mode": [secondary_mode],
                "quality_for_rn": [display_quality],
                "inversion_figure": [figure],
            }
        )
        result_df = split_degrees_to_single_degree(
            df,
            quality_col="quality_for_rn",
            inversion_col="inversion_figure",
            output_col="rn",
            inplace=False,
        )
        assert result_df["rn"].iloc[0] == expected_rn


# ── Class 4: semantic round-trip (requires music21) ─────────────────────


class TestSemanticRoundTrip:
    """Verify that split → rn preserves musical meaning (pitch classes).

    The rn string produced by split_degrees_to_single_degree with explicit
    quality is rnbert-style (uppercase degree + M/m suffix), so we use
    rn_format="rnbert" when resolving pitch classes.

    Requires music21 and mspell; skipped if unavailable.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_music21(self):
        pytest.importorskip("music21")
        pytest.importorskip("mspell")

    @pytest.mark.parametrize(
        "key, primary_degree, primary_alteration, secondary_degree, "
        "secondary_alteration, secondary_mode, raw_quality, inversion, expected_hex",
        [
            ("C", "I", "_", "I", "_", "_", "M", 0, "047"),
            ("C", "V", "_", "I", "_", "_", "M", 0, "7b2"),
            ("C", "I", "_", "I", "_", "_", "m", 0, "037"),
            ("G", "I", "_", "I", "_", "_", "M", 0, "7b2"),
            ("C", "V", "_", "V", "_", "_", "M", 0, "269"),
        ],
    )
    def test_pitch_classes_preserved(
        self,
        key,
        primary_degree,
        primary_alteration,
        secondary_degree,
        secondary_alteration,
        secondary_mode,
        raw_quality,
        inversion,
        expected_hex,
    ):
        from music_df.harmony.chords import get_rn_pitch_classes

        figure = inversion_number_to_figure(inversion, raw_quality)
        display_quality = get_quality_for_merging(raw_quality)
        df = pd.DataFrame(
            {
                "primary_degree": [primary_degree],
                "primary_alteration": [primary_alteration],
                "secondary_degree": [secondary_degree],
                "secondary_alteration": [secondary_alteration],
                "secondary_mode": [secondary_mode],
                "quality_for_rn": [display_quality],
                "inversion_figure": [figure],
            }
        )
        result_df = split_degrees_to_single_degree(
            df,
            quality_col="quality_for_rn",
            inversion_col="inversion_figure",
            output_col="rn",
            inplace=False,
        )
        rn_str = result_df["rn"].iloc[0]
        actual_hex = get_rn_pitch_classes(
            rn_str, key, hex_str=True, rn_format="rnbert"
        )
        assert actual_hex == expected_hex, (
            f"split->rn produced '{rn_str}'; "
            f"get_rn_pitch_classes('{rn_str}', '{key}', rn_format='rnbert') "
            f"= '{actual_hex}', expected '{expected_hex}'"
        )


# ── Class 5: rnbert → music21 ───────────────────────────────────────────


class TestRnbertToMusic21:
    """One-way translation tests. No music21 → rnbert exists yet
    (see notes/chord_df_formats.md Future Work).
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_music21(self):
        pytest.importorskip("music21")
        pytest.importorskip("mspell")

    @pytest.mark.parametrize(
        "rnbert, expected_music21",
        [
            ("IM", "I"),
            ("Im", "i"),
            ("VM", "V"),
            ("Vm", "v"),
            ("IVm6", "iv6"),
            ("VIIo7", "viio7"),
            ("VIIo642", "viio642"),
            ("III+", "III+"),
            ("bVIIM", "bVII"),
            # secondary
            ("VM/VM", "V/V"),
            ("VM/Vm", "V/v"),
            ("IVm/bIIm", "iv/bii"),
            ("VM/V", "V/V"),
            # augmented sixths
            ("xaug665", "Ger65"),
            ("xaug643", "Fr43"),
            ("xaug63", "It6"),
            ("xaug642", "It6"),
        ],
    )
    def test_translate(self, rnbert, expected_music21):
        from music_df.harmony.chords import translate_rns

        assert translate_rns(rnbert) == expected_music21


# ── Class 6: secondary_mode edge cases ──────────────────────────────────


class TestSecondaryModeEdgeCases:
    def test_underscore_no_secondary_joined_to_split(self):
        """'I' (no secondary) → split secondary_mode='_'."""
        df = _make_joined_df("I")
        result = single_degree_to_split_degrees(df, inplace=False)
        assert result["secondary_mode"].iloc[0] == "_"

    def test_underscore_with_secondary_joined_to_split(self):
        """'V/V' (no explicit mode) → split secondary_mode='_'."""
        df = _make_joined_df("V/V")
        result = single_degree_to_split_degrees(df, inplace=False)
        assert result["secondary_mode"].iloc[0] == "_"

    def test_mode_m_joined_to_split(self):
        """'V/Vm' → split secondary_mode='m'."""
        df = _make_joined_df("V/Vm")
        result = single_degree_to_split_degrees(df, inplace=False)
        assert result["secondary_degree"].iloc[0] == "V"
        assert result["secondary_mode"].iloc[0] == "m"

    def test_mode_M_joined_to_split(self):
        """'V/VM' → split secondary_mode='M'."""
        df = _make_joined_df("V/VM")
        result = single_degree_to_split_degrees(df, inplace=False)
        assert result["secondary_degree"].iloc[0] == "V"
        assert result["secondary_mode"].iloc[0] == "M"

    def test_mode_m_on_degree_I_round_trip(self):
        """secondary_mode='m' + secondary_degree='I' → '/Im' survives."""
        df = _make_split_df("V", "_", "I", "_", "m")
        joined = split_degrees_to_single_degree(df, inplace=False)
        assert joined["degree"].iloc[0] == "V/Im"
        result = single_degree_to_split_degrees(joined, inplace=False)
        assert result["secondary_degree"].iloc[0] == "I"
        assert result["secondary_mode"].iloc[0] == "m"

    def test_mode_M_on_degree_I_round_trip(self):
        """secondary_mode='M' + secondary_degree='I' → '/IM' survives."""
        df = _make_split_df("V", "_", "I", "_", "M")
        joined = split_degrees_to_single_degree(df, inplace=False)
        assert joined["degree"].iloc[0] == "V/IM"
        result = single_degree_to_split_degrees(joined, inplace=False)
        assert result["secondary_degree"].iloc[0] == "I"
        assert result["secondary_mode"].iloc[0] == "M"

    def test_mixed_secondary_modes(self):
        """DataFrame with varied secondary_mode values round-trips."""
        df = pd.DataFrame(
            {
                "primary_degree": ["V", "V", "V", "I"],
                "primary_alteration": ["_", "_", "_", "_"],
                "secondary_degree": ["V", "V", "I", "I"],
                "secondary_alteration": ["_", "_", "_", "_"],
                "secondary_mode": ["_", "m", "M", "_"],
            }
        )
        joined = split_degrees_to_single_degree(df, inplace=False)
        result = single_degree_to_split_degrees(joined, inplace=False)
        pd.testing.assert_series_equal(
            result["secondary_mode"],
            df["secondary_mode"],
            check_names=False,
        )

    def test_absent_secondary_mode_column(self):
        """When secondary_mode column is absent, split→joined still works
        and round-trip through single_degree_to_split_degrees creates it."""
        df = _make_split_df("V", "_", "V", "_")  # no secondary_mode
        assert "secondary_mode" not in df.columns
        joined = split_degrees_to_single_degree(df, inplace=False)
        assert joined["degree"].iloc[0] == "V/V"
        result = single_degree_to_split_degrees(joined, inplace=False)
        assert "secondary_mode" in result.columns
        assert result["secondary_mode"].iloc[0] == "_"


# ── Class 7: multi-row round-trips ──────────────────────────────────────


class TestMultiRowRoundTrips:
    def test_mixed_degrees_joined_to_split_round_trip(self):
        degrees = [
            "I",
            "bVI",
            "#IV",
            "V/V",
            "#VI/bII",
            "V/Vm",
            "V/VM",
        ]
        df = pd.DataFrame({"degree": degrees})
        split_df = single_degree_to_split_degrees(df, inplace=False)
        result_df = split_degrees_to_single_degree(split_df, inplace=False)
        assert list(result_df["degree"]) == degrees

    def test_mixed_split_to_joined_round_trip(self):
        df = pd.DataFrame(
            {
                "primary_degree": ["I", "V", "VII", "V"],
                "primary_alteration": ["_", "_", "#", "_"],
                "secondary_degree": ["I", "V", "V", "II"],
                "secondary_alteration": ["_", "_", "_", "b"],
                "secondary_mode": ["_", "m", "_", "M"],
            }
        )
        joined = split_degrees_to_single_degree(df, inplace=False)
        result = single_degree_to_split_degrees(joined, inplace=False)
        for col in [
            "primary_degree",
            "primary_alteration",
            "secondary_degree",
            "secondary_alteration",
            "secondary_mode",
        ]:
            pd.testing.assert_series_equal(
                result[col], df[col], check_names=False, obj=col
            )

    def test_with_type_column_and_null_chords(self):
        """Full round-trip with type column: non-note rows → null_chord_token,
        note rows round-trip exactly."""
        na = DEFAULT_NULL_CHORD_TOKEN
        df = pd.DataFrame(
            {
                "type": ["bar", "note", "note", "bar", "note"],
                "primary_degree": [na, "I", "V", na, "IV"],
                "primary_alteration": [na, "_", "_", na, "_"],
                "secondary_degree": [na, "I", "V", na, "I"],
                "secondary_alteration": [na, "_", "_", na, "_"],
                "secondary_mode": [na, "_", "m", na, "_"],
            }
        )
        joined = split_degrees_to_single_degree(df, inplace=False)
        assert joined.loc[0, "degree"] == na
        assert joined.loc[3, "degree"] == na
        assert joined.loc[1, "degree"] == "I"
        assert joined.loc[2, "degree"] == "V/Vm"
        assert joined.loc[4, "degree"] == "IV"

        # Full round-trip back to split
        result = single_degree_to_split_degrees(joined, inplace=False)
        for col in [
            "primary_degree",
            "primary_alteration",
            "secondary_degree",
            "secondary_alteration",
            "secondary_mode",
        ]:
            pd.testing.assert_series_equal(
                result[col], df[col], check_names=False, obj=col
            )
