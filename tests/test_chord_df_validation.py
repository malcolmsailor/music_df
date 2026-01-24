import pandas as pd
import pytest

from music_df.chord_df import (
    VALID_INVERSIONS,
    assert_valid_chord_df,
    validate_chord_df,
)


class TestValidateChordDfJoinedFormat:
    def test_valid_joined_format(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "C"],
                "degree": ["I", "V/V"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid
        assert result.format_detected == "joined"

    def test_valid_with_release(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "release": [1.0, 2.0],
                "key": ["C", "C"],
                "degree": ["I", "V"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid

    def test_valid_with_null_chord_token(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "na"],
                "degree": ["I", "na"],
                "quality": ["M", "na"],
                "inversion": [0, 0],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid

    def test_valid_secondary_degrees(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0],
                "key": ["C", "C", "C"],
                "degree": ["V/V", "#VII/bII", "bVI"],
                "quality": ["M", "d", "M"],
                "inversion": [0, 0, 0],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid


class TestValidateChordDfSplitFormat:
    def test_valid_split_format(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "C"],
                "primary_degree": ["I", "V"],
                "primary_alteration": ["_", "#"],
                "secondary_degree": ["I", "V"],
                "secondary_alteration": ["_", "_"],
                "quality": ["M", "M"],
                "inversion": [0, 1],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid
        assert result.format_detected == "split"


class TestValidateChordDfInvalidKey:
    def test_invalid_key(self):
        df = pd.DataFrame(
            {
                "onset": [0.0],
                "key": ["X"],
                "degree": ["I"],
                "quality": ["M"],
                "inversion": [0],
            }
        )
        result = validate_chord_df(df)
        assert not result.is_valid
        assert any("Invalid key" in e.message for e in result.errors)

    def test_multiple_invalid_keys(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["X", "Y"],
                "degree": ["I", "V"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            }
        )
        result = validate_chord_df(df)
        assert not result.is_valid


class TestValidateChordDfInvalidDegree:
    def test_invalid_degree_format(self):
        df = pd.DataFrame(
            {
                "onset": [0.0],
                "key": ["C"],
                "degree": ["not_a_degree"],
                "quality": ["M"],
                "inversion": [0],
            }
        )
        result = validate_chord_df(df)
        assert not result.is_valid
        assert any("Invalid degree" in e.message for e in result.errors)

    def test_invalid_degree_with_numbers(self):
        df = pd.DataFrame(
            {
                "onset": [0.0],
                "key": ["C"],
                "degree": ["I7"],  # quality shouldn't be in degree
                "quality": ["M"],
                "inversion": [0],
            }
        )
        result = validate_chord_df(df)
        assert not result.is_valid


class TestValidateChordDfInvalidInversion:
    def test_invalid_inversion_value(self):
        df = pd.DataFrame(
            {
                "onset": [0.0],
                "key": ["C"],
                "degree": ["I"],
                "quality": ["M"],
                "inversion": [5],
            }
        )
        result = validate_chord_df(df)
        assert not result.is_valid
        assert any("Invalid inversion" in e.message for e in result.errors)

    def test_valid_inversions(self):
        for inv in VALID_INVERSIONS:
            df = pd.DataFrame(
                {
                    "onset": [0.0],
                    "key": ["C"],
                    "degree": ["I"],
                    "quality": ["M"],
                    "inversion": [inv],
                }
            )
            result = validate_chord_df(df)
            assert result.is_valid, f"Inversion {inv} should be valid"


class TestValidateChordDfTemporalConsistency:
    def test_non_increasing_onsets_warning(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 2.0, 1.0],
                "key": ["C", "C", "C"],
                "degree": ["I", "IV", "V"],
                "quality": ["M", "M", "M"],
                "inversion": [0, 0, 0],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid  # warnings don't invalidate
        assert len(result.warnings) > 0
        assert any("non-decreasing" in w.message for w in result.warnings)

    def test_release_before_onset_warning(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "release": [0.5, 0.5],  # second release < onset
                "key": ["C", "C"],
                "degree": ["I", "V"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid
        assert any("Release < onset" in w.message for w in result.warnings)


class TestValidateChordDfStrictMode:
    def test_strict_mode_warnings_become_errors(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 2.0, 1.0],
                "key": ["C", "C", "C"],
                "degree": ["I", "IV", "V"],
                "quality": ["M", "M", "M"],
                "inversion": [0, 0, 0],
            }
        )
        result = validate_chord_df(df, strict=True)
        assert not result.is_valid
        assert len(result.warnings) == 0
        assert len(result.errors) > 0


class TestValidateChordDfMissingColumns:
    def test_missing_columns_unknown_format(self):
        df = pd.DataFrame(
            {
                "onset": [0.0],
                "key": ["C"],
            }
        )
        result = validate_chord_df(df)
        assert not result.is_valid
        assert result.format_detected == "unknown"

    def test_partial_joined_columns(self):
        df = pd.DataFrame(
            {
                "onset": [0.0],
                "key": ["C"],
                "degree": ["I"],
                # missing quality and inversion
            }
        )
        result = validate_chord_df(df)
        assert not result.is_valid
        assert result.format_detected == "unknown"


class TestValidateChordDfIndex:
    def test_valid_range_index(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "C"],
                "degree": ["I", "V"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            }
        )
        result = validate_chord_df(df)
        assert result.is_valid

    def test_invalid_index(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "C"],
                "degree": ["I", "V"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            },
            index=[1, 2],
        )  # not starting at 0
        result = validate_chord_df(df)
        assert not result.is_valid

    def test_skip_index_check(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "C"],
                "degree": ["I", "V"],
                "quality": ["M", "M"],
                "inversion": [0, 0],
            },
            index=[1, 2],
        )
        result = validate_chord_df(df, check_index=False)
        assert result.is_valid


class TestAssertValidChordDf:
    def test_raises_on_invalid(self):
        df = pd.DataFrame(
            {
                "onset": [0.0],
                "key": ["X"],
                "degree": ["I"],
                "quality": ["M"],
                "inversion": [0],
            }
        )
        with pytest.raises(ValueError, match="chord_df validation failed"):
            assert_valid_chord_df(df)

    def test_no_raise_on_valid(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "key": ["C", "G"],
                "degree": ["I", "V"],
                "quality": ["M", "M"],
                "inversion": [0, 1],
            }
        )
        assert_valid_chord_df(df)  # should not raise
