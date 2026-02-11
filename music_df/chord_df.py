import io  # noqa: F401
import re
from dataclasses import dataclass, field
from fractions import Fraction
from math import isnan
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from music_df.harmony.chords import (
    MAJOR_KEYS,
    MINOR_KEYS,
    CacheDict,
    get_key_pc_cache,
    get_rn_pc_cache,
)

# =============================================================================
# chord_df Specification
# =============================================================================

JOINED_FORMAT_REQUIRED_COLUMNS = frozenset(
    {"onset", "key", "degree", "quality", "inversion"}
)
SPLIT_FORMAT_REQUIRED_COLUMNS = frozenset(
    {
        "onset",
        "key",
        "primary_degree",
        "primary_alteration",
        "secondary_degree",
        "secondary_alteration",
        "quality",
        "inversion",
    }
)
OPTIONAL_COLUMNS = frozenset({"release", "chord_pcs"})
SPLIT_FORMAT_OPTIONAL_COLUMNS = frozenset({"secondary_mode"})

VALID_SECONDARY_MODES = frozenset({"m", "M"})

RN_FORMAT_REQUIRED_COLUMNS = frozenset({"onset", "key", "rn"})

VALID_ALTERATIONS = frozenset({"_", "#", "b", "##", "bb"})
VALID_INVERSIONS = frozenset({0, 1, 2, 3})
DEFAULT_NULL_CHORD_TOKEN = "na"

# Validates hex strings like "b26", "6a1", "047"
CHORD_PCS_REGEX = re.compile(r"^[0-9a-f]+$")

DEGREE_REGEX = re.compile(
    r"""
    ^
    ([#b]*)         # group 1: primary alteration (e.g., "#", "b", "##")
    ([IViv]+)       # group 2: primary Roman numeral (e.g., "I", "VII", "iv")
    (               # group 3: optional secondary (tonicization)
        /           #   literal slash separator
        ([#b]*)     #   group 4: secondary alteration
        ([IViv]+)   #   group 5: secondary Roman numeral
        ([mM]?)     #   group 6: optional secondary mode
    )?
    $
""",
    re.VERBOSE,
)

ENHARMONIC_KEY_REGEX = re.compile(r"^[A-Ga-g](#*|b*)$")


# =============================================================================
# Validation
# =============================================================================


@dataclass
class ValidationError:
    column: str | None
    row_index: int | None
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    format_detected: Literal["joined", "split", "rn", "unknown"] = "unknown"

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            error_messages = [f"  - {e.message}" for e in self.errors]
            raise ValueError(
                "chord_df validation failed:\n" + "\n".join(error_messages)
            )


def _detect_format(
    df: pd.DataFrame,
) -> Literal["joined", "split", "rn", "unknown"]:
    columns = set(df.columns)
    # Check joined and split first since they are supersets of rn
    if JOINED_FORMAT_REQUIRED_COLUMNS <= columns:
        return "joined"
    if SPLIT_FORMAT_REQUIRED_COLUMNS <= columns:
        return "split"
    if RN_FORMAT_REQUIRED_COLUMNS <= columns:
        return "rn"
    return "unknown"


def _check_index(df: pd.DataFrame, errors: list[ValidationError]) -> None:
    if isinstance(df.index, pd.RangeIndex):
        if df.index.start != 0 or df.index.stop != len(df) or df.index.step != 1:
            errors.append(
                ValidationError(
                    column=None,
                    row_index=None,
                    message=f"Index must be RangeIndex(0, {len(df)}, 1), got RangeIndex({df.index.start}, {df.index.stop}, {df.index.step})",
                )
            )
    elif not (df.index == range(len(df))).all():
        errors.append(
            ValidationError(
                column=None,
                row_index=None,
                message=f"Index must be equivalent to range(0, {len(df)})",
            )
        )


def _check_columns(
    df: pd.DataFrame,
    format_detected: Literal["joined", "split", "rn", "unknown"],
    errors: list[ValidationError],
) -> None:
    if format_detected == "unknown":
        errors.append(
            ValidationError(
                column=None,
                row_index=None,
                message=f"Could not detect format. Columns must include either {sorted(JOINED_FORMAT_REQUIRED_COLUMNS)} (joined), {sorted(SPLIT_FORMAT_REQUIRED_COLUMNS)} (split), or {sorted(RN_FORMAT_REQUIRED_COLUMNS)} (rn). Got: {sorted(df.columns)}",
            )
        )


def _is_fraction_dtype(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    non_null = series.dropna()
    return len(non_null) > 0 and all(isinstance(x, Fraction) for x in non_null)


def _check_types(
    df: pd.DataFrame,
    format_detected: Literal["joined", "split", "rn", "unknown"],
    errors: list[ValidationError],
) -> None:
    temporal_cols = ["onset"]
    if "release" in df.columns:
        temporal_cols.append("release")

    for col in temporal_cols:
        if col in df.columns:
            is_valid = pd.api.types.is_numeric_dtype(df[col]) or _is_fraction_dtype(
                df[col]
            )
            if not is_valid:
                errors.append(
                    ValidationError(
                        column=col,
                        row_index=None,
                        message=f"Column '{col}' must be numeric or Fraction, got {df[col].dtype}",
                    )
                )

    if "inversion" in df.columns and not pd.api.types.is_numeric_dtype(df["inversion"]):
        errors.append(
            ValidationError(
                column="inversion",
                row_index=None,
                message=f"Column 'inversion' must be numeric, got {df['inversion'].dtype}",
            )
        )

    if format_detected == "rn":
        # rn format only has key and rn as string columns
        string_cols = ["key", "rn"]
    else:
        string_cols = ["key", "quality"]
        if format_detected == "joined":
            string_cols.append("degree")
        elif format_detected == "split":
            string_cols.extend(
                [
                    "primary_degree",
                    "primary_alteration",
                    "secondary_degree",
                    "secondary_alteration",
                ]
            )
            if "secondary_mode" in df.columns:
                string_cols.append("secondary_mode")

    for col in string_cols:
        if col in df.columns and not pd.api.types.is_string_dtype(df[col]):
            errors.append(
                ValidationError(
                    column=col,
                    row_index=None,
                    message=f"Column '{col}' must be string type, got {df[col].dtype}",
                )
            )


def _check_key_values(
    df: pd.DataFrame,
    null_chord_token: str,
    errors: list[ValidationError],
    allow_enharmonic_keys: bool = False,
) -> None:
    if "key" not in df.columns:
        return

    non_null_mask = ~df["key"].isna()

    if allow_enharmonic_keys:

        def is_valid_key(val):
            return (
                val == null_chord_token
                or ENHARMONIC_KEY_REGEX.match(str(val)) is not None
            )

        invalid_mask = ~df["key"].apply(is_valid_key) & non_null_mask
    else:
        valid_keys = set(MAJOR_KEYS) | set(MINOR_KEYS) | {null_chord_token}
        invalid_mask = ~df["key"].isin(valid_keys) & non_null_mask

    if invalid_mask.any():
        invalid_indices = df.index[invalid_mask].tolist()
        invalid_values = df.loc[invalid_mask, "key"].unique().tolist()
        errors.append(
            ValidationError(
                column="key",
                row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                message=f"Invalid key value(s): {invalid_values} at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
            )
        )


def _check_degree_format(
    df: pd.DataFrame,
    null_chord_token: str,
    errors: list[ValidationError],
) -> None:
    if "degree" not in df.columns:
        return

    def is_valid_degree(val):
        if pd.isna(val) or val == null_chord_token:
            return True
        return DEGREE_REGEX.match(str(val)) is not None

    invalid_mask = ~df["degree"].apply(is_valid_degree)
    if invalid_mask.any():
        invalid_indices = df.index[invalid_mask].tolist()
        invalid_values = df.loc[invalid_mask, "degree"].unique().tolist()
        errors.append(
            ValidationError(
                column="degree",
                row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                message=f"Invalid degree format: {invalid_values} at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
            )
        )


def _check_alteration_values(
    df: pd.DataFrame,
    null_chord_token: str,
    null_alteration_char: str,
    errors: list[ValidationError],
) -> None:
    alteration_cols = ["primary_alteration", "secondary_alteration"]
    valid_alterations = VALID_ALTERATIONS | {null_chord_token, null_alteration_char}

    for col in alteration_cols:
        if col not in df.columns:
            continue
        invalid_mask = ~df[col].isin(valid_alterations) & ~df[col].isna()
        if invalid_mask.any():
            invalid_indices = df.index[invalid_mask].tolist()
            invalid_values = df.loc[invalid_mask, col].unique().tolist()
            errors.append(
                ValidationError(
                    column=col,
                    row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                    message=f"Invalid {col} value(s): {invalid_values} at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
                )
            )


def _check_secondary_mode_values(
    df: pd.DataFrame,
    null_chord_token: str,
    null_alteration_char: str,
    errors: list[ValidationError],
) -> None:
    if "secondary_mode" not in df.columns:
        return

    valid_values = VALID_SECONDARY_MODES | {null_alteration_char, null_chord_token}
    invalid_mask = (
        ~df["secondary_mode"].isin(valid_values) & ~df["secondary_mode"].isna()
    )
    if invalid_mask.any():
        invalid_indices = df.index[invalid_mask].tolist()
        invalid_values = df.loc[invalid_mask, "secondary_mode"].unique().tolist()
        errors.append(
            ValidationError(
                column="secondary_mode",
                row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                message=f"Invalid secondary_mode value(s): {invalid_values} at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
            )
        )


def _check_inversion_values(
    df: pd.DataFrame,
    errors: list[ValidationError],
) -> None:
    if "inversion" not in df.columns:
        return

    non_null_inversions = df["inversion"].dropna()
    if len(non_null_inversions) == 0:
        return

    invalid_mask = ~non_null_inversions.isin(VALID_INVERSIONS)
    if invalid_mask.any():
        invalid_indices = non_null_inversions.index[invalid_mask].tolist()
        invalid_values = non_null_inversions.loc[invalid_mask].unique().tolist()
        errors.append(
            ValidationError(
                column="inversion",
                row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                message=f"Invalid inversion value(s): {invalid_values} (must be in {sorted(VALID_INVERSIONS)}) at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
            )
        )


def _check_chord_pcs_values(
    df: pd.DataFrame,
    errors: list[ValidationError],
) -> None:
    """Validate chord_pcs column when present (hex string format)."""
    if "chord_pcs" not in df.columns:
        return

    # Check type: allow string dtype or object dtype (for mixed string/None)
    is_string = pd.api.types.is_string_dtype(df["chord_pcs"])
    is_object = df["chord_pcs"].dtype == object
    if not (is_string or is_object):
        errors.append(
            ValidationError(
                column="chord_pcs",
                row_index=None,
                message=f"Column 'chord_pcs' must be string type, got {df['chord_pcs'].dtype}",
            )
        )
        return

    # Check values: must be hex digits only (null and empty allowed)
    def is_valid_chord_pcs(val):
        if pd.isna(val) or val == "":
            return True
        if not isinstance(val, str):
            return False
        return CHORD_PCS_REGEX.match(val) is not None

    invalid_mask = ~df["chord_pcs"].apply(is_valid_chord_pcs)
    if invalid_mask.any():
        invalid_indices = df.index[invalid_mask].tolist()
        invalid_values = df.loc[invalid_mask, "chord_pcs"].unique().tolist()
        errors.append(
            ValidationError(
                column="chord_pcs",
                row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                message=f"Invalid chord_pcs value(s): {invalid_values} (must be hex digits) at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
            )
        )


def _check_key_values_rn_format(
    df: pd.DataFrame,
    null_chord_token: str,
    errors: list[ValidationError],
    allow_enharmonic_keys: bool = False,
) -> None:
    """Validate key column for rn format (allows empty strings for ffill)."""
    if "key" not in df.columns:
        return

    # First row key must not be empty (required for ffill)
    first_key = df["key"].iloc[0] if len(df) > 0 else None
    if first_key is None or pd.isna(first_key) or first_key == "":
        errors.append(
            ValidationError(
                column="key",
                row_index=0,
                message="First row key must not be empty (required for forward-fill)",
            )
        )

    # Non-empty keys must be valid
    non_empty_mask = ~df["key"].isna() & (df["key"] != "")

    if allow_enharmonic_keys:

        def is_valid_key(val):
            return (
                val == null_chord_token
                or ENHARMONIC_KEY_REGEX.match(str(val)) is not None
            )

        invalid_mask = ~df["key"].apply(is_valid_key) & non_empty_mask
    else:
        valid_keys = set(MAJOR_KEYS) | set(MINOR_KEYS) | {null_chord_token}
        invalid_mask = ~df["key"].isin(valid_keys) & non_empty_mask

    if invalid_mask.any():
        invalid_indices = df.index[invalid_mask].tolist()
        invalid_values = df.loc[invalid_mask, "key"].unique().tolist()
        errors.append(
            ValidationError(
                column="key",
                row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                message=f"Invalid key value(s): {invalid_values} at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
            )
        )


def _check_unrecognized_columns(
    df: pd.DataFrame,
    format_detected: Literal["joined", "split", "rn", "unknown"],
    errors: list[ValidationError],
) -> None:
    if format_detected == "joined":
        recognized = JOINED_FORMAT_REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    elif format_detected == "split":
        recognized = (
            SPLIT_FORMAT_REQUIRED_COLUMNS
            | SPLIT_FORMAT_OPTIONAL_COLUMNS
            | OPTIONAL_COLUMNS
        )
    elif format_detected == "rn":
        recognized = RN_FORMAT_REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    else:
        recognized = (
            JOINED_FORMAT_REQUIRED_COLUMNS
            | SPLIT_FORMAT_REQUIRED_COLUMNS
            | RN_FORMAT_REQUIRED_COLUMNS
            | OPTIONAL_COLUMNS
        )

    unrecognized = set(df.columns) - recognized
    if unrecognized:
        errors.append(
            ValidationError(
                column=None,
                row_index=None,
                message=f"Unrecognized column(s): {sorted(unrecognized)}",
            )
        )


def _check_temporal_consistency(
    df: pd.DataFrame,
    warnings: list[ValidationError],
) -> None:
    if "onset" not in df.columns:
        return

    onset_diff = df["onset"].diff()
    decreasing_mask = onset_diff < 0
    if decreasing_mask.any():
        decreasing_indices = df.index[decreasing_mask].tolist()
        warnings.append(
            ValidationError(
                column="onset",
                row_index=decreasing_indices[0]
                if len(decreasing_indices) == 1
                else None,
                message=f"Onset values are not non-decreasing at row(s) {decreasing_indices[:5]}{'...' if len(decreasing_indices) > 5 else ''}",
                severity="warning",
            )
        )

    if "release" in df.columns:
        invalid_release_mask = df["release"] < df["onset"]
        invalid_release_mask = invalid_release_mask & ~df["release"].isna()
        if invalid_release_mask.any():
            invalid_indices = df.index[invalid_release_mask].tolist()
            warnings.append(
                ValidationError(
                    column="release",
                    row_index=invalid_indices[0] if len(invalid_indices) == 1 else None,
                    message=f"Release < onset at row(s) {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}",
                    severity="warning",
                )
            )


def validate_chord_df(
    chord_df: pd.DataFrame,
    *,
    format: Literal["joined", "split", "rn", "auto"] = "auto",
    null_chord_token: str = DEFAULT_NULL_CHORD_TOKEN,
    null_alteration_char: str = "_",
    strict: bool = False,
    check_index: bool = True,
    check_values: bool = True,
    check_types: bool = True,
    allow_enharmonic_keys: bool = False,
) -> ValidationResult:
    """
    Validate a chord_df DataFrame.

    Parameters
    ----------
    chord_df : pd.DataFrame
        The DataFrame to validate.
    format : {"joined", "split", "rn", "auto"}
        The expected format. If "auto", detect from columns.
    null_chord_token : str
        Token used for null/rest chords (default "na").
    null_alteration_char : str
        Character used for no alteration in split format (default "_").
    strict : bool
        If True, treat warnings as errors and fail on unrecognized columns.
    check_index : bool
        If True, validate that index is a RangeIndex starting at 0.
    check_values : bool
        If True, validate column values (keys, degrees, inversions, etc.).
    check_types : bool
        If True, validate column types.
    allow_enharmonic_keys : bool
        If True, accept any key matching the pattern [A-Ga-g](#*|b*) instead of
        validating against the predefined MAJOR_KEYS and MINOR_KEYS lists.

    Returns
    -------
    ValidationResult
        Object containing validation results.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "onset": [0.0, 1.0],
    ...         "key": ["C", "C"],
    ...         "degree": ["I", "V/V"],
    ...         "quality": ["M", "M"],
    ...         "inversion": [0, 0],
    ...     }
    ... )
    >>> result = validate_chord_df(df)
    >>> result.is_valid
    True
    >>> result.format_detected
    'joined'

    >>> df_bad = pd.DataFrame(
    ...     {
    ...         "onset": [0.0],
    ...         "key": ["X"],
    ...         "degree": ["I"],
    ...         "quality": ["M"],
    ...         "inversion": [0],
    ...     }
    ... )
    >>> result = validate_chord_df(df_bad)
    >>> result.is_valid
    False
    >>> "Invalid key" in result.errors[0].message
    True
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # Detect format
    if format == "auto":
        format_detected = _detect_format(chord_df)
    else:
        format_detected = format

    # Check index
    if check_index:
        _check_index(chord_df, errors)

    # Check columns
    _check_columns(chord_df, format_detected, errors)

    # Check types
    if check_types and format_detected != "unknown":
        _check_types(chord_df, format_detected, errors)

    # Check values
    if check_values and format_detected != "unknown":
        if format_detected == "rn":
            _check_key_values_rn_format(
                chord_df, null_chord_token, errors, allow_enharmonic_keys
            )
            # TODO(validation): rn content validation could be added in the future.
            # Currently skipped because rn format varies by cache (e.g., "V" vs "VM").
        else:
            _check_key_values(chord_df, null_chord_token, errors, allow_enharmonic_keys)

            if format_detected == "joined":
                _check_degree_format(chord_df, null_chord_token, errors)
            else:
                _check_alteration_values(
                    chord_df, null_chord_token, null_alteration_char, errors
                )
                _check_secondary_mode_values(
                    chord_df, null_chord_token, null_alteration_char, errors
                )

            _check_inversion_values(chord_df, errors)

        _check_temporal_consistency(chord_df, warnings)
        _check_chord_pcs_values(chord_df, errors)

    if strict:
        _check_unrecognized_columns(chord_df, format_detected, errors)
        errors.extend(warnings)
        warnings = []

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        format_detected=format_detected,
    )


def assert_valid_chord_df(chord_df: pd.DataFrame, **kwargs) -> None:
    """
    Validate a chord_df and raise ValueError if invalid.

    Parameters
    ----------
    chord_df : pd.DataFrame
        The DataFrame to validate.
    **kwargs
        Arguments passed to validate_chord_df.

    Raises
    ------
    ValueError
        If the chord_df is invalid.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "onset": [0.0, 1.0],
    ...         "key": ["C", "G"],
    ...         "degree": ["I", "V"],
    ...         "quality": ["M", "M"],
    ...         "inversion": [0, 1],
    ...     }
    ... )
    >>> assert_valid_chord_df(df)  # No error

    >>> df_bad = pd.DataFrame(
    ...     {
    ...         "onset": [0.0],
    ...         "key": ["X"],
    ...         "degree": ["I"],
    ...         "quality": ["M"],
    ...         "inversion": [0],
    ...     }
    ... )
    >>> assert_valid_chord_df(df_bad)
    Traceback (most recent call last):
        ...
    ValueError: chord_df validation failed:
      - Invalid key value(s): ['X'] at row(s) [0]
    """
    result = validate_chord_df(chord_df, **kwargs)
    result.raise_if_invalid()


def add_chord_pcs(
    chord_df: pd.DataFrame,
    inplace: bool = False,
    rn_pc_cache: CacheDict[tuple[str, str], list[int] | str] | None = None,
) -> pd.DataFrame:
    """
    Adds a column 'chord_pcs' to the chord_df with the pcs of each chord.

    >>> chord_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... key,rn
    ... b,i
    ... ,V
    ... f#,iv
    ... ,i
    ... ,V/V
    ... '''
    ...     )
    ... )
    >>> add_chord_pcs(chord_df)
      key   rn chord_pcs
    0   b    i       b26
    1   b    V       6a1
    2  f#   iv       b26
    3  f#    i       691
    4  f#  V/V       803

    >>> chord_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... key,rn
    ... b,Im
    ... ,VM
    ... f#,IVm
    ... ,Im
    ... ,VM/V
    ... '''
    ...     )
    ... )
    >>> add_chord_pcs(
    ...     chord_df,
    ...     rn_pc_cache=get_rn_pc_cache(rn_format="rnbert", hex_str=True),
    ... )
      key    rn chord_pcs
    0   b    Im       b26
    1   b    VM       6a1
    2  f#   IVm       b26
    3  f#    Im       691
    4  f#  VM/V       803
    """
    if not inplace:
        chord_df = chord_df.copy()

    chord_df["key"] = chord_df["key"].ffill()

    if rn_pc_cache is None:
        rn_pc_cache = get_rn_pc_cache(case_matters=True, hex_str=True)

    chord_df["chord_pcs"] = chord_df.apply(
        lambda row: rn_pc_cache[row["rn"], row["key"]],
        axis=1,
    )

    return chord_df


def add_key_pcs(
    key_df: pd.DataFrame,
    inplace: bool = False,
    key_pc_cache: CacheDict[str, list[int] | str] | None = None,
) -> pd.DataFrame:
    """
    Adds a column 'key_pcs' to the df with the pcs of each key.

    >>> key_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... key
    ... b
    ... NaN
    ... f#
    ... '''
    ...     )
    ... )
    >>> add_key_pcs(key_df)
      key  key_pcs
    0   b  b12467a
    1   b  b12467a
    2  f#  689b125
    """
    if not inplace:
        key_df = key_df.copy()

    key_df["key"] = key_df["key"].ffill()

    if key_pc_cache is None:
        key_pc_cache = get_key_pc_cache(hex_str=True)

    key_df["key_pcs"] = key_df["key"].apply(lambda key: key_pc_cache[key])

    return key_df


def get_quality_for_merging(quality: pd.Series | str) -> pd.Series | str:
    # (Malcolm 2024-04-18) possibly we want to do further processing, e.g.
    #   - remove 6 from augmented 6 chords "aug6" quality and otherwise simplify
    #   - only display the quality when it contradicts the expected value for the
    #       scale (this of course would require a lot more coding)
    if isinstance(quality, str):
        return quality.replace("7", "")
    else:
        return quality.str.replace("7", "")


def keep_new_elements_only(
    series: pd.Series, fill_element="", ignore_falsy: bool = True
):
    """
    >>> s = pd.Series(list("aaabbcddde"))
    >>> keep_new_elements_only(s)  # doctest: +NORMALIZE_WHITESPACE
    0    a
    1
    2
    3    b
    4
    5    c
    6    d
    7
    8
    9    e
    dtype: object

    >>> s = pd.Series([float("nan"), "a", "", "a", float("nan"), "b", ""])
    >>> keep_new_elements_only(s)  # doctest: +NORMALIZE_WHITESPACE
    0
    1    a
    2
    3
    4
    5    b
    6
    dtype: object
    """

    out = series.copy()
    if ignore_falsy:
        out[~series.astype(bool)] = float("nan")
    out = out.ffill()

    mask = (out != out.shift(1)) & (~out.isna())
    out[~mask] = fill_element
    return out


def merge_annotations(
    df: pd.DataFrame,
    degree_col: str = "degree",
    primary_degree_col: str = "primary_degree",
    primary_alteration_col: str = "primary_alteration",
    secondary_degree_col: str = "secondary_degree",
    secondary_alteration_col: str = "secondary_alteration",
    secondary_mode_col: str = "secondary_mode",
    inversion_col: str = "inversion",
    quality_col: str = "quality",
    include_key: bool = True,
    key_col: str = "key",
) -> pd.Series:
    """
    We rely on the dataframe being sorted to only show new annotations.

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,key,degree,inversion,quality
    ... bar,,,,
    ... note,C,I,0,M
    ... note,C,V,1,M
    ... bar,na,na,,na
    ... note,C,I,0,M
    ... note,G,V,1,M
    ... '''
    ...     )
    ... )
    >>> merge_annotations(df)  # doctest: +NORMALIZE_WHITESPACE
    0
    1     C.IM
    2      VM6
    3
    4       IM
    5    G.VM6
    dtype: object

    Testing that V in C is not the same as V in a.
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,key,degree,inversion,quality
    ... bar,,,,
    ... note,a,V,0,M
    ... note,C,V,0,M
    ... '''
    ...     )
    ... )
    >>> merge_annotations(df)  # doctest: +NORMALIZE_WHITESPACE
    0
    1    a.VM
    2    C.VM
    dtype: object
    >>> merge_annotations(df, include_key=False)  # doctest: +NORMALIZE_WHITESPACE
    0
    1    VM
    2    VM
    dtype: object
    """
    df = df.copy()

    df["inversion_figure"] = df.apply(
        lambda row: inversion_number_to_figure(row[inversion_col], row[quality_col]),
        axis=1,
    )
    df["quality_for_merging"] = get_quality_for_merging(df[quality_col])
    if degree_col not in df.columns:
        assert all(
            col in df.columns
            for col in [
                primary_degree_col,
                primary_alteration_col,
                secondary_degree_col,
                secondary_alteration_col,
            ]
        )
    else:
        df = single_degree_to_split_degrees(
            df,
            degree_col=degree_col,
            inplace=True,
            primary_degree_col=primary_degree_col,
            primary_alteration_col=primary_alteration_col,
            secondary_degree_col=secondary_degree_col,
            secondary_alteration_col=secondary_alteration_col,
            secondary_mode_col=secondary_mode_col,
        )
    df = split_degrees_to_single_degree(
        df,
        inversion_col="inversion_figure",
        quality_col="quality_for_merging",
        primary_degree_col=primary_degree_col,
        primary_alteration_col=primary_alteration_col,
        secondary_degree_col=secondary_degree_col,
        secondary_alteration_col=secondary_alteration_col,
        secondary_mode_col=secondary_mode_col,
        output_col="rn",
        inplace=True,
    )
    # I was using ":" as the separator character but it is a special
    #   value in humdrum even when escaped.

    rn_with_nan = df["rn"].replace("na", float("nan"))
    df["rn"] = keep_new_elements_only(rn_with_nan)
    keys = keep_new_elements_only(df[key_col].replace("na", float("nan")) + ".")

    # When key is shown but rn was filtered out (same as previous), restore rn
    key_shown = keys.astype(bool)
    rn_empty = df["rn"] == ""
    rn_available = ~rn_with_nan.ffill().isna()
    restore_mask = key_shown & rn_empty & rn_available
    df.loc[restore_mask, "rn"] = rn_with_nan.ffill()[restore_mask]

    if include_key:
        return keys + df["rn"]
    return pd.Series(df["rn"].values)


def get_unique_annotations_per_onset(
    df: pd.DataFrame,
    annotation_col: str,
    onset_col: str = "onset",
    pitch_col: str = "pitch",
    fill_value: str = "",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Rather than using this function, we can just sort the dataframe and then use
    keep_new_elements_only. This has the virtue of only showing each annotation when
    it first occurs rather than on every new onset.

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,pitch,rn
    ... 0.0,60,IM
    ... 0.0,64,IM
    ... 0.5,62,IM
    ... 1.0,67,VM6
    ... 1.0,59,VM6
    ... '''
    ...     )
    ... )
    >>> get_unique_annotations_per_onset(df, "rn")  # doctest: +NORMALIZE_WHITESPACE
       onset  pitch   rn
    0    0.0     60   IM
    1    0.0     64
    2    0.5     62   IM
    3    1.0     67
    4    1.0     59  VM6
    """
    if not inplace:
        df = df.copy()

    assert annotation_col in df.columns, f"annotation_col {annotation_col} not in df"

    min_pitch_indices = df.groupby(onset_col)[pitch_col].idxmin()
    mask = df.index.isin(min_pitch_indices)
    df.loc[~mask, annotation_col] = fill_value
    return df


TRIAD_INVERSIONS = MappingProxyType({0: "", 1: "6", 2: "64"})
SEVENTH_CHORD_INVERSIONS = MappingProxyType({0: "7", 1: "65", 2: "43", 3: "42"})


def inversion_number_to_figure(
    inversion_number: int,
    quality: str,
    triad_inversions_mapping: Mapping[int, str] | None = None,
    seventh_chord_inversions_mapping: Mapping[int, str] | None = None,
) -> str:
    """
    Convert a number indicating the inversion (0-indexed) to a figured-bass figure.

    Quality is required to distinguish between triads on the one hand and 7th chords and
    augmented 6ths on the other.

    >>> inversion_number_to_figure(0, "M")
    ''
    >>> inversion_number_to_figure(1, "m")
    '6'
    >>> inversion_number_to_figure(0, "m7")
    '7'
    >>> inversion_number_to_figure(3, "Mm7")
    '42'
    """
    # If the chord is a 7th or augmented 6th, we use 7th chord inversions. (Since
    #   we only have integers to indicate 1st, 2nd inversion etc., we can't distinguish
    #   German and Italian 6th chords.)
    temp_inversion_number = float(inversion_number)
    if isnan(temp_inversion_number):
        return ""
    inversion_number = int(temp_inversion_number)
    if "7" in quality or quality == "aug6":
        if seventh_chord_inversions_mapping is None:
            seventh_chord_inversions_mapping = SEVENTH_CHORD_INVERSIONS
        return seventh_chord_inversions_mapping.get(inversion_number, "?")
    # If the quality is unknown we ignore the inversion
    elif quality == "x":
        return ""
    # Otherwise, assume to be a triad
    if triad_inversions_mapping is None:
        triad_inversions_mapping = TRIAD_INVERSIONS
    return triad_inversions_mapping.get(inversion_number, "?")


def split_degrees_to_single_degree(
    df: pd.DataFrame,
    primary_degree_col: str = "primary_degree",
    primary_alteration_col: str = "primary_alteration",
    secondary_degree_col: str = "secondary_degree",
    secondary_alteration_col: str = "secondary_alteration",
    secondary_mode_col: str = "secondary_mode",
    inversion_col: str | None = None,
    quality_col: str | None = None,
    null_alteration_char: str = "_",
    output_col: str = "degree",
    null_chord_token: str = "na",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,primary_degree,primary_alteration,secondary_degree,secondary_alteration,quality,inversion
    ... bar,na,na,na,na,na,na
    ... note,I,_,I,_,M,
    ... note,VII,#,V,_,d,43
    ... bar,,,,
    ... note,V,_,VII,b,M,6
    ... note,VI,#,II,b,m,64
    ... note,VI,b,I,_,M,
    ... '''
    ...     )
    ... )
    >>> split_degrees_to_single_degree(df)["degree"]
    0         na
    1          I
    2     #VII/V
    3         na
    4     V/bVII
    5    #VI/bII
    6        bVI
    Name: degree, dtype: object
    >>> split_degrees_to_single_degree(
    ...     df, inversion_col="inversion", quality_col="quality", output_col="rn"
    ... )["rn"]
    0            na
    1            IM
    2     #VIId43/V
    3            na
    4      VM6/bVII
    5    #VIm64/bII
    6          bVIM
    Name: rn, dtype: object

    With secondary_mode column:
    >>> df2 = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... primary_degree,primary_alteration,secondary_degree,secondary_alteration,secondary_mode
    ... I,_,I,_,_
    ... V,_,V,_,m
    ... V,_,V,_,M
    ... VII,#,V,_,m
    ... VI,b,I,_,_
    ... '''
    ...     )
    ... )
    >>> split_degrees_to_single_degree(df2)["degree"]
    0          I
    1       V/Vm
    2       V/VM
    3    #VII/Vm
    4        bVI
    Name: degree, dtype: object
    """
    if not inplace:
        df = df.copy()

    df[output_col] = (
        df[primary_alteration_col]
        + df[primary_degree_col]
        + (df[quality_col].fillna("") if quality_col is not None else "")
        + (df[inversion_col].fillna("") if inversion_col is not None else "")
        + "/"
        + df[secondary_alteration_col]
        + df[secondary_degree_col]
        + (
            df[secondary_mode_col].fillna("")
            if secondary_mode_col in df.columns
            else ""
        )
    )
    df[output_col] = df[output_col].str.replace(null_alteration_char, "")

    # Remove "/I" from the end of Roman numerals
    df[output_col] = df[output_col].str.replace(r"/I$", "", regex=True)

    if "type" in df.columns:
        df.loc[df["type"] != "note", output_col] = null_chord_token

    return df


def single_degree_to_split_degrees(
    df: pd.DataFrame,
    degree_col: str = "degree",
    primary_degree_col: str = "primary_degree",
    primary_alteration_col: str = "primary_alteration",
    secondary_degree_col: str = "secondary_degree",
    secondary_alteration_col: str = "secondary_alteration",
    secondary_mode_col: str = "secondary_mode",
    null_alteration_char: str = "_",
    null_chord_token: str = "na",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... degree
    ... I
    ... IV
    ... V
    ... I
    ... '''
    ...     )
    ... )
    >>> single_degree_to_split_degrees(df)  # doctest: +NORMALIZE_WHITESPACE
      degree primary_degree primary_alteration secondary_degree secondary_alteration secondary_mode
    0      I              I                  _                I                    _              _
    1     IV             IV                  _                I                    _              _
    2      V              V                  _                I                    _              _
    3      I              I                  _                I                    _              _

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... degree
    ... I
    ... VII/V
    ... V/bVII
    ... #VI/bII
    ... bVI
    ... na
    ... '''
    ...     )
    ... )
    >>> single_degree_to_split_degrees(df)  # doctest: +NORMALIZE_WHITESPACE
        degree primary_degree primary_alteration secondary_degree secondary_alteration secondary_mode
    0        I              I                  _                I                    _              _
    1    VII/V            VII                  _                V                    _              _
    2   V/bVII              V                  _              VII                    b              _
    3  #VI/bII             VI                  #               II                    b              _
    4      bVI             VI                  b                I                    _              _
    5       na             na                 na               na                   na             na

    With secondary mode suffixes:
    >>> df2 = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... degree
    ... V/Vm
    ... V/VM
    ... VII/bIIm
    ... I
    ... na
    ... '''
    ...     )
    ... )
    >>> single_degree_to_split_degrees(df2)  # doctest: +NORMALIZE_WHITESPACE
         degree primary_degree primary_alteration secondary_degree secondary_alteration secondary_mode
    0      V/Vm              V                  _                V                    _              m
    1      V/VM              V                  _                V                    _              M
    2  VII/bIIm            VII                  _               II                    b              m
    3         I              I                  _                I                    _              _
    4        na             na                 na               na                   na             na
    """
    if not inplace:
        df = df.copy()

    splits = df[degree_col].str.split("/", n=1, expand=True)

    null_mask = (df[degree_col] == null_chord_token) | (df[degree_col].isna())

    primary = (
        splits[0]
        .str.extract(r"([b#]*)(.*)")
        .rename(columns={0: primary_alteration_col, 1: primary_degree_col})
    )
    primary[primary_alteration_col] = (
        primary[primary_alteration_col]
        .fillna(null_alteration_char)
        .replace("", null_alteration_char)
    )
    primary.loc[null_mask, :] = null_chord_token
    df[primary_degree_col] = primary[primary_degree_col]
    df[primary_alteration_col] = primary[primary_alteration_col]

    if splits.shape[1] == 1:
        # There are no secondary degrees
        df[secondary_degree_col] = "I"
        df[secondary_alteration_col] = null_alteration_char
        df[secondary_mode_col] = null_alteration_char

    else:
        secondary = (
            splits[1]
            .str.extract(r"([b#]*)([IViv]+)([mM]?)")
            .rename(
                columns={
                    0: secondary_alteration_col,
                    1: secondary_degree_col,
                    2: secondary_mode_col,
                }
            )
        )

        secondary[secondary_alteration_col] = (
            secondary[secondary_alteration_col]
            .fillna(null_alteration_char)
            .replace("", null_alteration_char)
        )
        secondary[secondary_degree_col] = secondary[secondary_degree_col].fillna("I")
        secondary[secondary_mode_col] = (
            secondary[secondary_mode_col]
            .fillna(null_alteration_char)
            .replace("", null_alteration_char)
        )

        secondary.loc[null_mask, :] = null_chord_token

        df[secondary_degree_col] = secondary[secondary_degree_col]
        df[secondary_alteration_col] = secondary[secondary_alteration_col]
        df[secondary_mode_col] = secondary[secondary_mode_col]

    return df


def drop_harmony_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If, for example, we are re-labeling a score, we want to drop all columns that may
    have chord annotations to be sure they don't leak through.
    """
    patterns = [
        "key",
        "degree",
        "mode",
        "rn",
        "chord",
        "harmony",
        "bass",
        "alteration",
        "inversion",
        "quality",
        "root",
    ]
    harmony_cols = [
        c for c in df.columns if any(pattern in c.lower() for pattern in patterns)
    ]
    # print(f"Removing columns: {harmony_cols}")
    df = df.drop(columns=harmony_cols)
    # print(f"Remaining columns: {df.columns}")
    return df


def extract_chord_df_from_music_df(
    music_df: pd.DataFrame,
    null_chord_token: str = "na",
    columns: Iterable[str] = ("key", "onset", "degree", "quality", "inversion"),
    release_col: str = "release",
) -> pd.DataFrame:
    """

    >>> music_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,key,onset,release,degree,quality,inversion
    ... bar,,,0.0,4.0,,,
    ... note,60,C,0.0,1.0,I,M,0.0
    ... note,64,C,1.0,2.0,I,M,0.0
    ... note,62,C,2.0,3.0,V,M,1.0
    ... note,67,C,3.0,4.0,V,M,0.0
    ... note,66,G,4.0,6.0,V,M,0.0
    ... '''
    ...     )
    ... )
    >>> extract_chord_df_from_music_df(music_df)
      key  onset degree quality  inversion  release
    0   C    0.0      I       M        0.0      2.0
    1   C    2.0      V       M        1.0      3.0
    2   C    3.0      V       M        0.0      4.0
    3   G    4.0      V       M        0.0      6.0
    """
    import warnings

    columns = list(columns)
    missing = [c for c in columns if c not in music_df.columns]
    if missing:
        warnings.warn(
            f"Columns not found in music_df (skipping): {missing}",
            stacklevel=2,
        )
        columns = [c for c in columns if c in music_df.columns]
    assert all(col in music_df.columns for col in columns), (
        f"music_df must have the following columns: {columns}"
    )

    chord_change_masks = [
        music_df[col] != music_df[col].shift(1) for col in columns if col != "onset"
    ]
    chord_change_mask = np.logical_or.reduce(chord_change_masks)

    chord_df = music_df.loc[chord_change_mask, columns].copy()
    chord_df = chord_df.loc[
        (chord_df[columns[0]] != null_chord_token) & (~chord_df[columns[0]].isna())
    ]
    chord_df = chord_df.reset_index(drop=True)

    if release_col in music_df.columns:
        chord_df[release_col] = chord_df["onset"].shift(-1)
        # We use .max() rather than .iloc[-1] because the last item in music_df may
        # not have a release time, e.g. if it is a bar.
        chord_df.loc[len(chord_df) - 1, release_col] = music_df[release_col].max()

    return chord_df


def extract_key_df_from_music_df(
    music_df_or_chord_df: pd.DataFrame,
    null_key_token: str = "na",
    columns: Iterable[str] = ("key", "onset"),
    release_col: str = "release",
) -> pd.DataFrame:
    """
    This is really just a simple wrapper for extract_chord_df_from_music_df with
    different default parameters.

    >>> music_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,key,onset,release,degree,quality,inversion
    ... bar,,,0.0,4.0,,,
    ... note,60,C,0.0,1.0,I,M,0.0
    ... note,64,C,1.0,2.0,I,M,0.0
    ... note,62,C,2.0,3.0,V,M,1.0
    ... note,67,C,3.0,4.0,V,M,0.0
    ... note,66,G,4.0,6.0,V,M,0.0
    ... note,65,F,8.0,10.0,V,M,0.0
    ... '''
    ...     )
    ... )
    >>> extract_key_df_from_music_df(music_df)
      key  onset  release
    0   C    0.0      4.0
    1   G    4.0      8.0
    2   F    8.0     10.0

    It should work on chord_df as well as music_df:
    >>> chord_df = extract_chord_df_from_music_df(music_df)
    >>> extract_key_df_from_music_df(chord_df)
      key  onset  release
    0   C    0.0      4.0
    1   G    4.0      8.0
    2   F    8.0     10.0
    """
    return extract_chord_df_from_music_df(
        music_df_or_chord_df,
        null_chord_token=null_key_token,
        columns=columns,
        release_col=release_col,
    )


def label_music_df_with_chord_df(
    music_df: pd.DataFrame,
    chord_df: pd.DataFrame,
    columns_to_add: Iterable[str] = ("key", "degree", "quality", "inversion"),
    null_chord_token: str = "na",
) -> pd.DataFrame:
    """
    >>> music_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,1.0
    ... note,64,1.0,2.0
    ... note,62,2.0,3.0
    ... note,67,3.0,4.0
    ... bar,,4.0,8.0
    ... note,66,4.0,6.0
    ... note,67,6.0,8.0
    ... '''
    ...     )
    ... )
    >>> chord_df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... onset,key,degree,quality,inversion
    ... 0.0,C,I,M,0.0
    ... 3.0,C,V,M,1.0
    ... 5.0,G,V,M,0.0
    ... 7.0,G,I,M,0.0
    ... '''
    ...     )
    ... )
    >>> label_music_df_with_chord_df(music_df, chord_df)
       type  pitch  onset  release key degree quality  inversion
    0   bar    NaN    0.0      4.0  na     na      na        NaN
    1  note   60.0    0.0      1.0   C      I       M        0.0
    2  note   64.0    1.0      2.0   C      I       M        0.0
    3  note   62.0    2.0      3.0   C      I       M        0.0
    4  note   67.0    3.0      4.0   C      V       M        1.0
    5   bar    NaN    4.0      8.0  na     na      na        NaN
    6  note   66.0    4.0      6.0   C      V       M        1.0
    7  note   67.0    6.0      8.0   G      V       M        0.0
    """
    import warnings

    columns_to_add = list(columns_to_add)
    missing = [c for c in columns_to_add if c not in chord_df.columns]
    if missing:
        warnings.warn(
            f"Columns not found in chord_df (skipping): {missing}",
            stacklevel=2,
        )
        columns_to_add = [c for c in columns_to_add if c in chord_df.columns]

    out = pd.merge_asof(
        music_df.drop(columns=[c for c in columns_to_add if c in music_df.columns]),
        chord_df[["onset"] + columns_to_add],
        on="onset",
        direction="backward",
    )
    nonnote_mask = out["type"] != "note"
    for col in columns_to_add:
        if out[col].dtype == "object":
            out.loc[nonnote_mask, col] = null_chord_token
        else:
            out.loc[nonnote_mask, col] = float("nan")

    return out
