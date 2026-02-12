"""Demo: piano roll with highlighted regions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from music_df.plot import plot_piano_roll
from music_df.read_csv import read_csv

EXAMPLE_DIR = Path("tests/resources/example_dfs")

if __name__ == "__main__":
    # --- Load notes and chords for the Bach chorale ---
    notes_df = read_csv(
        EXAMPLE_DIR / "Bach,_Johann_Sebastian_Chorales_010_notes.csv"
    ).reset_index()
    chords_df = pd.read_csv(
        EXAMPLE_DIR / "Bach,_Johann_Sebastian_Chorales_010_chords.csv",
        dtype={"chord_pcs": str},
    )

    annotations = [
        (row.onset, f"{row.key}.{row.primary_degree}{row.quality}")
        for _, row in chords_df.iterrows()
    ]

    excerpt = notes_df[
        (notes_df.type.isin(["note", "bar"])) & (notes_df.onset < 16)
    ].reset_index(drop=True)
    excerpt_annotations = [(o, l) for o, l in annotations if o < 16]

    regions = [
        (0.0, 4.0),  # full-height: first 4 beats
        (8.0, 12.0, 60.0, 67.0),  # bounded: beats 8-12, mid-range pitches
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_piano_roll(
        excerpt,
        annotations=excerpt_annotations,
        regions=regions,
        barlines=True,
        ax=ax,
        title="Bach Chorale 010 — with regions",
    )
    plt.show()
