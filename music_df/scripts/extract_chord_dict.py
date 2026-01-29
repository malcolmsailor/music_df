"""
Extract unique roman numerals from CSV files and generate a YAML chord dictionary
organized by mode → quality → degree.
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
]


INVERSION_SUFFIXES = re.compile(r"(6|64|7|65|43|42)$")

SPLIT_DEGREE_REQUIRED_COLUMNS = frozenset(
    {
        "key",
        "primary_degree",
        "primary_alteration",
        "secondary_degree",
        "secondary_alteration",
        "quality",
    }
)
DEGREE_REQUIRED_COLUMNS = frozenset({"key", "degree", "quality"})
RN_REQUIRED_COLUMNS = frozenset({"key", "rn"})


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


def strip_inversion(rn: str) -> str:
    """Strip inversion suffixes from a roman numeral."""
    return INVERSION_SUFFIXES.sub("", rn)


def build_rn_from_split_degree(row: pd.Series, null_alt_char: str = "_") -> str:
    """Build RN string from split-degree columns."""
    primary_alt = row["primary_alteration"]
    primary_deg = row["primary_degree"]
    quality = row["quality"]
    secondary_alt = row["secondary_alteration"]
    secondary_deg = row["secondary_degree"]

    if primary_alt == null_alt_char:
        primary_alt = ""
    if secondary_alt == null_alt_char:
        secondary_alt = ""

    rn = f"{primary_alt}{primary_deg}{quality}"
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


def extract_chord_dict(
    csv_files: list[Path], rn_format: str, *, show_progress: bool = False
) -> tuple[dict[str, Any], list[FailedChord]]:
    """
    Extract chord dictionary from CSV files.

    Returns:
        - nested dict with field_names header and hierarchy:
          mode_name → quality → primary_degree → primary_alteration →
          secondary_degree → secondary_alteration → hex_pitch_classes
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

    # Deduplicate by all 6 hierarchy fields while keeping one example of each
    seen: dict[tuple[str, str, str, str, str, str], ExtractedChord] = {}
    for chord in all_chords:
        key = (
            chord.mode,
            chord.quality,
            chord.primary_degree,
            chord.primary_alteration,
            chord.secondary_degree,
            chord.secondary_alteration,
        )
        if key not in seen:
            seen[key] = chord

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

        mode_name = MODE_INTERNAL_TO_YAML[chord.mode]

        # Build nested hierarchy: mode → quality → primary_degree →
        # primary_alteration → secondary_degree → secondary_alteration
        if mode_name not in output:
            output[mode_name] = {}
        quality_dict = output[mode_name]

        if chord.quality not in quality_dict:
            quality_dict[chord.quality] = {}
        primary_degree_dict = quality_dict[chord.quality]

        if chord.primary_degree not in primary_degree_dict:
            primary_degree_dict[chord.primary_degree] = {}
        primary_alt_dict = primary_degree_dict[chord.primary_degree]

        if chord.primary_alteration not in primary_alt_dict:
            primary_alt_dict[chord.primary_alteration] = {}
        secondary_degree_dict = primary_alt_dict[chord.primary_alteration]

        if chord.secondary_degree not in secondary_degree_dict:
            secondary_degree_dict[chord.secondary_degree] = {}
        secondary_alt_dict = secondary_degree_dict[chord.secondary_degree]

        secondary_alt_dict[chord.secondary_alteration] = pcs

    return {"field_names": SPLIT_DEGREE_FIELD_NAMES, **output}, failures


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
    """Integration test: CSV input → hierarchical YAML output."""
    import tempfile

    csv_content = """\
key,primary_degree,primary_alteration,secondary_degree,secondary_alteration,quality
C,I,_,I,_,M
C,V,_,I,_,M
C,V,_,IV,_,Mm7
C,I,_,I,_,+
a,I,_,I,_,m
"""
    expected = {
        "field_names": [
            "mode",
            "quality",
            "primary_degree",
            "primary_alteration",
            "secondary_degree",
            "secondary_alteration",
        ],
        "major": {
            "+": {
                "I": {"_": {"I": {"_": "048"}}},
            },
            "M": {
                "I": {"_": {"I": {"_": "047"}}},
                "V": {"_": {"I": {"_": "7b2"}}},
            },
            "Mm7": {
                "V": {"_": {"IV": {"_": "047a"}}},
            },
        },
        "minor": {
            "m": {
                "I": {"_": {"I": {"_": "037"}}},
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        result, failures = extract_chord_dict([csv_path], "split-degree")
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
    args = parser.parse_args()

    csv_files = find_csv_files(args.input_paths)
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    chord_dict, failures = extract_chord_dict(
        csv_files, args.rn_format, show_progress=True
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
        chord_dict, Dumper=dumper, default_flow_style=False, sort_keys=False
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_str)
        print(f"Wrote output to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_str)


if __name__ == "__main__":
    main()
