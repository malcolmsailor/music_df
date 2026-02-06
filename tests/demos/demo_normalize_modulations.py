"""Demo: fzf-select a CSV, normalize modulations, plot piano roll with
tonicization regions boxed and key changes annotated."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from music_df.chord_df import (
    extract_chord_df_from_music_df,
    single_degree_to_split_degrees,
    split_degrees_to_single_degree,
)
from music_df.harmony.modulation import (
    remove_long_tonicizations,
    remove_short_modulations,
)
from music_df.plot import plot_piano_roll
from music_df.read_csv import read_csv

DEFAULT_DEMO_DIR = Path("tests/resources/example_dfs")


def select_file_with_fzf(csv_dir: Path) -> tuple[str, bool]:
    """Present deduplicated base names via fzf. Returns (selection, is_paired).

    Selection is a path relative to csv_dir (e.g. "subdir/Bach_010" for paired,
    "subdir/standalone.csv" for standalone).
    """
    csvs = sorted(csv_dir.rglob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {csv_dir}")
        sys.exit(1)

    bases: dict[str, bool] = {}
    standalone: list[str] = []
    for p in csvs:
        rel = p.relative_to(csv_dir)
        name = p.stem
        rel_dir = rel.parent
        if name.endswith("_notes"):
            base = name[: -len("_notes")]
            chords_path = p.parent / f"{base}_chords.csv"
            if chords_path.exists():
                bases[str(rel_dir / base)] = True
            else:
                standalone.append(str(rel))
        elif name.endswith("_chords"):
            base = name[: -len("_chords")]
            notes_path = p.parent / f"{base}_notes.csv"
            if notes_path.exists():
                bases.setdefault(str(rel_dir / base), True)
            else:
                standalone.append(str(rel))
        else:
            standalone.append(str(rel))

    choices = sorted(bases.keys()) + sorted(standalone)
    input_text = "\n".join(choices)

    try:
        result = subprocess.run(
            ["fzf"],
            input=input_text,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(1)

    selection = result.stdout.strip()
    if not selection:
        sys.exit(1)

    is_paired = selection in bases
    return selection, is_paired


def load_data(
    selected: str, is_paired: bool, csv_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load notes and chord DataFrames. Returns chord_df in split format."""
    if is_paired:
        notes_path = csv_dir / f"{selected}_notes.csv"
        chords_path = csv_dir / f"{selected}_chords.csv"
        assert notes_path.exists(), f"Notes file not found: {notes_path}"
        assert chords_path.exists(), f"Chords file not found: {chords_path}"
        notes_df = read_csv(notes_path).reset_index()
        chord_df = pd.read_csv(chords_path, dtype={"chord_pcs": str})
        if "secondary_degree" not in chord_df.columns:
            chord_df = single_degree_to_split_degrees(chord_df)
    else:
        path = csv_dir / selected
        assert path.exists(), f"CSV file not found: {path}"
        music_df = read_csv(path).reset_index()
        notes_df = music_df
        music_df_joined = split_degrees_to_single_degree(music_df, inplace=False)
        chord_df = extract_chord_df_from_music_df(music_df_joined)
        chord_df = single_degree_to_split_degrees(chord_df)

    return notes_df, chord_df


def find_tonicization_regions(
    chord_df: pd.DataFrame,
) -> list[tuple[float, float]]:
    """Find consecutive runs where secondary_degree != 'I', return (onset, release)."""
    if "secondary_degree" not in chord_df.columns:
        chord_df = single_degree_to_split_degrees(chord_df, inplace=False)

    secondary = chord_df["secondary_degree"].values
    onsets = chord_df["onset"].values
    releases = (
        chord_df["release"].values if "release" in chord_df.columns else None
    )

    regions: list[tuple[float, float]] = []
    i = 0
    n = len(chord_df)
    while i < n:
        if secondary[i] != "I":
            start = i
            while i < n and secondary[i] != "I":
                i += 1
            end = i - 1
            region_onset = float(onsets[start])
            if releases is not None and not pd.isna(releases[end]):
                region_release = float(releases[end])
            elif end + 1 < n:
                region_release = float(onsets[end + 1])
            else:
                region_release = float(onsets[end])
            regions.append((region_onset, region_release))
        else:
            i += 1

    return regions


def find_key_annotations(
    chord_df: pd.DataFrame,
) -> list[tuple[float, str]]:
    """Emit (onset, key) wherever key changes from previous row. Always include first."""
    annotations: list[tuple[float, str]] = []
    prev_key = None
    for _, row in chord_df.iterrows():
        key = row["key"]
        if key != prev_key:
            annotations.append((float(row["onset"]), str(key)))
            prev_key = key
    return annotations


def emit_notation_pdf(
    notes_df: pd.DataFrame, chord_df: pd.DataFrame, pdf_path: str
) -> None:
    """Render notes + chord annotations to a PDF via Humdrum."""
    from music_df.humdrum_export.humdrum_export import df_with_harmony_to_hum
    from music_df.humdrum_export.pdf import run_hum2pdf

    split_degree = "secondary_degree" in chord_df.columns
    humdrum = df_with_harmony_to_hum(
        notes_df, chord_df, split_degree=split_degree
    )
    with tempfile.NamedTemporaryFile(
        suffix=".krn", mode="w", delete=False
    ) as f:
        f.write(humdrum)
        krn_path = f.name
    try:
        run_hum2pdf(krn_path, pdf_path)
    finally:
        os.remove(krn_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit-notation",
        action="store_true",
        help="Save annotated PDFs (original + normalized) to ~/tmp",
    )
    args = parser.parse_args()

    csv_dir = Path(os.environ.get("DEMO_CSVS", str(DEFAULT_DEMO_DIR)))
    selected, is_paired = select_file_with_fzf(csv_dir)
    notes_df, chord_df = load_data(selected, is_paired, csv_dir)

    if "secondary_degree" not in chord_df.columns:
        chord_df = single_degree_to_split_degrees(chord_df, inplace=False)

    orig_regions = find_tonicization_regions(chord_df)
    orig_annotations = find_key_annotations(chord_df)

    norm_chord_df = remove_long_tonicizations(
        chord_df,
        max_tonicization_duration=16.0,
        min_removal_duration=4.0,
        max_tonicization_num_chords=2,
        min_removal_num_chords=2,
    )
    norm_chord_df = remove_short_modulations(
        norm_chord_df,
        min_modulation_duration=2.0,
        max_removal_duration=16.0,
        min_modulation_num_chords=3,
        max_removal_num_chords=8,
    )

    if "secondary_degree" not in norm_chord_df.columns:
        norm_chord_df = single_degree_to_split_degrees(
            norm_chord_df, inplace=False
        )

    norm_regions = find_tonicization_regions(norm_chord_df)
    norm_annotations = find_key_annotations(norm_chord_df)

    if args.emit_notation:
        output_dir = Path.home() / "tmp"
        output_dir.mkdir(exist_ok=True)
        base_name = Path(selected).name
        emit_notation_pdf(
            notes_df, chord_df, str(output_dir / f"{base_name}_original.pdf")
        )
        emit_notation_pdf(
            notes_df,
            norm_chord_df,
            str(output_dir / f"{base_name}_normalized.pdf"),
        )
        print(f"PDFs written to {output_dir}")

    fig, (ax_orig, ax_norm) = plt.subplots(2, 1, figsize=(14, 9))
    plot_piano_roll(
        notes_df,
        regions=orig_regions,
        annotations=orig_annotations,
        barlines=True,
        ax=ax_orig,
        title=f"{selected} — original",
        show=False,
    )
    plot_piano_roll(
        notes_df,
        regions=norm_regions,
        annotations=norm_annotations,
        barlines=True,
        ax=ax_norm,
        title=f"{selected} — normalized modulations",
        show=False,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
