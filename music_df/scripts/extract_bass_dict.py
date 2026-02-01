"""
Extract unique roman numerals from CSV files and generate a YAML bass dictionary
organized by mode → quality → degree → inversion, with bass pitch class (0-11) as values.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from music_df.chord_df import inversion_number_to_figure
from music_df.harmony.chords import get_rn_pitch_classes
from music_df.keys import get_mode

MODE_INTERNAL_TO_YAML = {"M": "major", "m": "minor"}


@dataclass
class ExtractedChord:
    mode: str
    quality: str
    primary_degree: str
    primary_alteration: str
    secondary_degree: str
    secondary_alteration: str
    inversion: str
    rn: str
    source_file: Path
    row_data: dict[str, Any]


SPLIT_DEGREE_FIELD_NAMES = [
    "mode",
    "quality",
    "primary_degree",
    "primary_alteration",
    "secondary_degree",
    "secondary_alteration",
    "inversion",
]

FieldStructure = list[list[str] | str]


def parse_concatenate_spec(
    spec: str | None, all_fields: list[str]
) -> FieldStructure | None:
    """
    Parse --concatenate spec and return new field structure.

    Returns a list where each element is either:
    - A single field name (str)
    - A list of field names to concatenate (list[str])

    Validates that all original fields appear exactly once.
    Returns None if spec is None.

    >>> parse_concatenate_spec(None, ["a", "b", "c"])
    >>> parse_concatenate_spec("a+b", ["a", "b", "c"])
    [['a', 'b'], 'c']
    >>> parse_concatenate_spec("b+c", ["a", "b", "c"])
    ['a', ['b', 'c']]
    """
    if spec is None:
        return None

    groups = spec.split(",")
    concatenated_fields: list[list[str]] = []
    all_specified: set[str] = set()

    for group in groups:
        fields = group.split("+")
        if len(fields) < 2:
            raise ValueError(
                f"Concatenation group must have at least 2 fields: {group!r}"
            )
        for field in fields:
            if field not in all_fields:
                raise ValueError(f"Unknown field: {field!r}")
            if field in all_specified:
                raise ValueError(f"Field appears more than once: {field!r}")
            all_specified.add(field)
        concatenated_fields.append(fields)

    # Build structure: iterate through original fields, replacing with groups
    structure: FieldStructure = []
    used_groups: set[int] = set()

    for field in all_fields:
        if field in all_specified:
            # Find which group this field belongs to
            for i, group in enumerate(concatenated_fields):
                if field in group and i not in used_groups:
                    structure.append(group)
                    used_groups.add(i)
                    break
        else:
            structure.append(field)

    return structure


def get_field_names_from_structure(structure: FieldStructure | None) -> list[str]:
    """
    Convert structure to field_names list (joining concatenated fields with _).

    >>> get_field_names_from_structure(None)
    ['mode', 'quality', 'primary_degree', 'primary_alteration', 'secondary_degree', 'secondary_alteration', 'inversion']
    >>> get_field_names_from_structure(["a", ["b", "c"], "d"])
    ['a', 'b_c', 'd']
    """
    if structure is None:
        return SPLIT_DEGREE_FIELD_NAMES.copy()

    result: list[str] = []
    for item in structure:
        if isinstance(item, list):
            result.append("_".join(item))
        else:
            result.append(item)
    return result


INVERSION_SUFFIX_PATTERN = re.compile(r"(6|64|7|65|43|42)$")

SPLIT_DEGREE_REQUIRED_COLUMNS = frozenset(
    {
        "key",
        "primary_degree",
        "primary_alteration",
        "secondary_degree",
        "secondary_alteration",
        "quality",
        "inversion",
    }
)
DEGREE_REQUIRED_COLUMNS = frozenset({"key", "degree", "quality", "inversion"})
RN_REQUIRED_COLUMNS = frozenset({"key", "rn", "inversion"})


def find_csv_files(paths: list[Path]) -> list[Path]:
    """Find all CSV files from the given paths (files or directories)."""
    csv_files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".csv":
            csv_files.append(path)
        elif path.is_dir():
            csv_files.extend(path.rglob("*.csv"))
    return csv_files


def has_required_columns(df: pd.DataFrame, required: frozenset[str]) -> bool:
    """Check if the DataFrame has the required columns."""
    return required <= set(df.columns)


def build_rn_from_split_degree(row: pd.Series, null_alt_char: str = "_") -> str:
    """Build RN string from split-degree columns, including inversion suffix.

    The RN format is: alteration + degree + quality + inversion + /secondary
    e.g., "bIIM64", "VMm765/IV"
    """
    primary_alt = row["primary_alteration"]
    primary_deg = row["primary_degree"]
    quality = row["quality"]
    secondary_alt = row["secondary_alteration"]
    secondary_deg = row["secondary_degree"]
    inversion = row["inversion"]

    if primary_alt == null_alt_char:
        primary_alt = ""
    if secondary_alt == null_alt_char:
        secondary_alt = ""
    if inversion == null_alt_char:
        inversion = ""
    else:
        inversion = inversion_number_to_figure(int(inversion), quality)

    # Format: alteration + degree + quality + inversion
    rn = f"{primary_alt}{primary_deg}{quality}{inversion}"

    if secondary_deg != "I":
        rn = f"{rn}/{secondary_alt}{secondary_deg}"
    return rn


def extract_from_split_degree(
    df: pd.DataFrame, source_file: Path
) -> list[ExtractedChord] | None:
    """
    Extract ExtractedChord objects from split-degree format.

    Returns list of ExtractedChord, or None if missing columns.
    """
    if not has_required_columns(df, SPLIT_DEGREE_REQUIRED_COLUMNS):
        return None

    results: list[ExtractedChord] = []
    for _, row in df.iterrows():
        key = row["key"]
        if pd.isna(key) or key == "":
            continue

        mode = get_mode(key)
        quality = row["quality"]
        if pd.isna(quality) or quality == "na":
            continue

        rn = build_rn_from_split_degree(row)

        results.append(
            ExtractedChord(
                mode=mode,
                quality=quality,
                primary_degree=row["primary_degree"],
                primary_alteration=row["primary_alteration"],
                secondary_degree=row["secondary_degree"],
                secondary_alteration=row["secondary_alteration"],
                inversion=row["inversion"],
                rn=rn,
                source_file=source_file,
                row_data=row.to_dict(),
            )
        )

    return results


def extract_from_degree(
    df: pd.DataFrame, source_file: Path
) -> list[ExtractedChord] | None:
    """
    Extract ExtractedChord objects from degree format.

    Returns list of ExtractedChord, or None if missing columns.
    """
    raise NotImplementedError("degree format not yet implemented")


def extract_from_rn(df: pd.DataFrame, source_file: Path) -> list[ExtractedChord] | None:
    """
    Extract ExtractedChord objects from rn format.

    Quality is extracted from the RN string itself.
    Returns list of ExtractedChord, or None if missing columns.
    """
    raise NotImplementedError("rn format not yet implemented")


EXTRACTORS = {
    "split-degree": extract_from_split_degree,
    "degree": extract_from_degree,
    "rn": extract_from_rn,
}


@dataclass
class FailedChord:
    rn: str
    mode: str
    error: str
    source_file: Path
    row_data: dict[str, Any]


EXPECTED_FAILURES = {"xx/x"}


def _get_chord_field_value(chord: ExtractedChord, field: str) -> str:
    """Get a field value from an ExtractedChord, with mode converted to YAML name."""
    if field == "mode":
        return MODE_INTERNAL_TO_YAML[chord.mode]
    return getattr(chord, field)


def _get_hierarchy_key(chord: ExtractedChord, structure_item: list[str] | str) -> str:
    """Get the hierarchy key for a chord given a structure item."""
    if isinstance(structure_item, list):
        return "".join(_get_chord_field_value(chord, f) for f in structure_item)
    return _get_chord_field_value(chord, structure_item)


def extract_bass_dict(
    csv_files: list[Path],
    rn_format: str,
    *,
    show_progress: bool = False,
    field_structure: FieldStructure | None = None,
) -> tuple[dict[str, Any], list[FailedChord]]:
    """
    Extract bass dictionary from CSV files.

    Args:
        csv_files: List of CSV file paths to process.
        rn_format: Format of roman numerals in CSV.
        show_progress: Whether to show progress bars.
        field_structure: Optional structure for hierarchy levels. Each element is
            either a single field name (str) or a list of field names to concatenate.
            If None, uses default SPLIT_DEGREE_FIELD_NAMES structure.

    Returns:
        - nested dict with field_names header and hierarchy:
          mode_name → quality → primary_degree → primary_alteration →
          secondary_degree → secondary_alteration → inversion → bass_pitch_class (0-11)
        - list of FailedChord for rows that couldn't be processed
    """
    extractor = EXTRACTORS[rn_format]

    all_chords: list[ExtractedChord] = []
    skipped_files: list[Path] = []
    file_iter = tqdm(csv_files, desc="Reading CSVs", disable=not show_progress)
    for csv_file in file_iter:
        df = pd.read_csv(csv_file)
        chords = extractor(df, csv_file)
        if chords is None:
            skipped_files.append(csv_file)
            continue
        all_chords.extend(chords)

    if skipped_files:
        print(
            f"Skipped {len(skipped_files)} file(s) missing required columns.",
            file=sys.stderr,
        )

    # Deduplicate by all 7 hierarchy fields while keeping one example of each
    seen: dict[tuple[str, str, str, str, str, str, str], ExtractedChord] = {}
    for chord in all_chords:
        key = (
            chord.mode,
            chord.quality,
            chord.primary_degree,
            chord.primary_alteration,
            chord.secondary_degree,
            chord.secondary_alteration,
            chord.inversion,
        )
        if key not in seen:
            seen[key] = chord

    # Use default structure if none provided
    if field_structure is None:
        structure: FieldStructure = list(SPLIT_DEGREE_FIELD_NAMES)
    else:
        structure = field_structure

    output: dict[str, Any] = {}
    failures: list[FailedChord] = []

    chord_iter = tqdm(
        sorted(seen.keys()), desc="Computing pitch classes", disable=not show_progress
    )
    for key in chord_iter:
        chord = seen[key]

        try:
            pcs = get_rn_pitch_classes(
                chord.rn, chord.mode, hex_str=True, rn_format="rnbert"
            )
        except Exception as e:
            if chord.rn not in EXPECTED_FAILURES:
                print(
                    f"Warning: Could not compute pitch classes for {chord.rn} in {chord.mode}: {e}",
                    file=sys.stderr,
                )
                failures.append(
                    FailedChord(
                        rn=chord.rn,
                        mode=chord.mode,
                        error=str(e),
                        source_file=chord.source_file,
                        row_data=chord.row_data,
                    )
                )
            continue

        # Extract bass pitch class (first hex digit → int 0-11)
        bass = int(pcs[0], 16)

        # Build nested hierarchy dynamically based on structure
        current_dict = output
        for i, structure_item in enumerate(structure):
            hierarchy_key = _get_hierarchy_key(chord, structure_item)
            is_last = i == len(structure) - 1
            if is_last:
                current_dict[hierarchy_key] = bass
            else:
                if hierarchy_key not in current_dict:
                    current_dict[hierarchy_key] = {}
                current_dict = current_dict[hierarchy_key]

    field_names = get_field_names_from_structure(field_structure)
    return {"field_names": field_names, **output}, failures


def _quoted_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


def _diff_nested_dicts(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Return list of differences between nested dicts with their paths."""
    diffs: list[str] = []

    if type(expected) is not type(actual):
        diffs.append(
            f"{path}: type {type(expected).__name__} vs {type(actual).__name__}"
        )
        return diffs

    if isinstance(expected, dict):
        all_keys = set(expected.keys()) | set(actual.keys())
        for key in sorted(all_keys, key=str):
            key_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                diffs.append(f"{key_path}: missing (expected {expected[key]!r})")
            elif key not in expected:
                diffs.append(f"{key_path}: unexpected (got {actual[key]!r})")
            else:
                diffs.extend(_diff_nested_dicts(expected[key], actual[key], key_path))
    elif expected != actual:
        diffs.append(f"{path}: {expected!r} != {actual!r}")

    return diffs


def test_split_degree_hierarchy() -> None:
    """Integration test: CSV input → hierarchical YAML output with bass pitch classes."""
    import tempfile

    csv_content = """\
key,primary_degree,primary_alteration,secondary_degree,secondary_alteration,quality,inversion
C,I,_,I,_,M,_
C,I,_,I,_,M,6
C,I,_,I,_,M,64
C,V,_,I,_,M,_
C,V,_,IV,_,Mm7,_
C,V,_,IV,_,Mm7,65
C,I,_,I,_,+,_
a,I,_,I,_,m,_
a,I,_,I,_,m,6
"""
    # Pitch classes are relative to the tonic, not absolute
    # C major IM = (0 4 7), bass = 0
    # C major I6M = (4 7 0), bass = 4
    # C major I64M = (7 0 4), bass = 7
    # C major VM = (7 b 2), bass = 7
    # C major VMm7/IV = (0 4 7 a), bass = 0
    # C major V65Mm7/IV = (0 a 7 9), bass = 0  (reordered by inversion)
    # C major I+ = (0 4 8), bass = 0
    # A minor Im = (0 3 7), bass = 0  (relative to A)
    # A minor I6m = (3 7 0), bass = 3  (relative to A)
    expected = {
        "field_names": [
            "mode",
            "quality",
            "primary_degree",
            "primary_alteration",
            "secondary_degree",
            "secondary_alteration",
            "inversion",
        ],
        "major": {
            "+": {
                "I": {"_": {"I": {"_": {"_": 0}}}},
            },
            "M": {
                "I": {"_": {"I": {"_": {"_": 0, "6": 4, "64": 7}}}},
                "V": {"_": {"I": {"_": {"_": 7}}}},
            },
            "Mm7": {
                "V": {"_": {"IV": {"_": {"_": 0, "65": 0}}}},
            },
        },
        "minor": {
            "m": {
                "I": {"_": {"I": {"_": {"_": 0, "6": 3}}}},
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        result, failures = extract_bass_dict([csv_path], "split-degree")
        diffs = _diff_nested_dicts(expected, result)
        assert not diffs, "Differences:\n" + "\n".join(diffs)
        assert failures == []
    finally:
        csv_path.unlink()


def main() -> None:
    if "--test" in sys.argv:
        import pytest

        sys.exit(pytest.main([__file__, "-v"]))

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="CSV files or directories to process (directories are searched recursively)",
    )
    parser.add_argument(
        "--rn-format",
        required=True,
        choices=["split-degree", "degree", "rn"],
        help=(
            "How roman numerals are represented in CSV: "
            "split-degree (uses primary_degree, primary_alteration, etc.), "
            "degree (uses degree, quality columns), "
            "rn (uses only rn column, quality extracted from RN string)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output YAML path (default: stdout)",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=None,
        help="Path to save CSV of rows that failed pitch class extraction",
    )
    parser.add_argument(
        "--concatenate",
        type=str,
        default=None,
        help=(
            "Concatenate fields into single hierarchy levels. "
            "Format: field1+field2+field3,field4+field5 (comma separates groups, + joins fields)"
        ),
    )
    args = parser.parse_args()

    csv_files = find_csv_files(args.input_paths)
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    field_structure = parse_concatenate_spec(args.concatenate, SPLIT_DEGREE_FIELD_NAMES)

    bass_dict, failures = extract_bass_dict(
        csv_files, args.rn_format, show_progress=True, field_structure=field_structure
    )

    if failures and args.debug_output:
        args.debug_output.parent.mkdir(parents=True, exist_ok=True)
        debug_rows = []
        for f in failures:
            debug_row = {
                "source_file": str(f.source_file),
                "rn": f.rn,
                "mode": f.mode,
                "error": f.error,
                **{f"row_{k}": v for k, v in f.row_data.items()},
            }
            debug_rows.append(debug_row)
        pd.DataFrame(debug_rows).to_csv(args.debug_output, index=False)
        print(
            f"Wrote {len(failures)} failed row(s) to {args.debug_output}",
            file=sys.stderr,
        )

    dumper = yaml.Dumper
    dumper.add_representer(str, _quoted_str_representer)

    output_str = yaml.dump(
        bass_dict, Dumper=dumper, default_flow_style=False, sort_keys=False
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_str)
        print(f"Wrote output to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_str)


if __name__ == "__main__":
    main()
