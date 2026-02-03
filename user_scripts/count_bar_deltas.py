"""
Count bar deltas (differences in bar number between consecutive notes) across a
directory of music files.

Usage:
    python user_scripts/count_bar_deltas.py <directory> --output <output_folder> \
        [--max-files N] [--seed S] [--workers W]
"""

import argparse
import os
import random
from collections import Counter
from glob import glob
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from music_df.add_feature import add_default_time_sig, infer_barlines, make_bar_explicit
from music_df.read import read

ANOMALY_THRESHOLD = 10


def process_file(file_path: str) -> tuple[Counter[int], str, list[tuple[int, int]]] | None:
    """Process a single file and return bar delta counts and file path."""
    try:
        df = read(file_path)

        # Check if bar events exist; if not, add time signature and infer barlines
        if not (df.type == "bar").any():
            df = add_default_time_sig(df)
            df = infer_barlines(df)

        df = make_bar_explicit(df)

        notes = df[df.type == "note"]
        if len(notes) < 2:
            return Counter(), file_path, []

        bar_deltas = notes.bar_number.diff().iloc[1:]
        anomaly_mask = bar_deltas.abs() >= ANOMALY_THRESHOLD
        anomalies = [(idx, int(delta)) for idx, delta in bar_deltas[anomaly_mask].items()]
        return Counter(bar_deltas.astype(int).tolist()), file_path, anomalies
    except Exception as e:
        print(f"Warning: failed to process {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Count bar deltas across a directory of music files."
    )
    parser.add_argument("directory", help="Directory containing music files")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument(
        "--max-files", type=int, default=None, help="Maximum number of files to process"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for file sampling"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    args = parser.parse_args()

    extensions = ("*.csv", "*.mid", "*.midi")
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(args.directory, "**", ext), recursive=True))

    if not files:
        print(f"No music files found in {args.directory}")
        return

    if args.max_files is not None and args.max_files < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.max_files)

    num_workers = args.workers if args.workers is not None else cpu_count()

    total_counts: Counter[int] = Counter()
    processed_files: list[str] = []
    all_anomalies: list[tuple[str, int, int]] = []

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_file, files),
                total=len(files),
                desc="Processing files",
            )
        )

    for result in results:
        if result is not None:
            counts, file_path, anomalies = result
            total_counts.update(counts)
            processed_files.append(file_path)
            for row_index, delta in anomalies:
                all_anomalies.append((file_path, row_index, delta))

    os.makedirs(args.output, exist_ok=True)

    counts_path = os.path.join(args.output, "bar_delta_counts.csv")
    with open(counts_path, "w") as f:
        f.write("bar_delta,count\n")
        for bar_delta, count in sorted(total_counts.items()):
            f.write(f"{bar_delta},{count}\n")

    files_path = os.path.join(args.output, "processed_files.txt")
    with open(files_path, "w") as f:
        for file_path in sorted(processed_files):
            f.write(f"{file_path}\n")

    anomalies_path = os.path.join(args.output, "anomalies.txt")
    with open(anomalies_path, "w") as f:
        for file_path, row_index, delta in sorted(all_anomalies):
            f.write(f"{file_path}:{row_index} (delta={delta})\n")

    print(f"Processed {len(processed_files)} files")
    print(f"Bar delta counts saved to {counts_path}")
    print(f"Processed files list saved to {files_path}")
    print(f"Anomalies ({len(all_anomalies)} total) saved to {anomalies_path}")


if __name__ == "__main__":
    main()
