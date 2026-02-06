"""Demo: piano roll with chord annotations below the x-axis."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from music_df.read_csv import read_csv
from music_df.plot import plot_piano_roll

EXAMPLE_DIR = Path("tests/resources/example_dfs")

# --- Load notes and chords for the Bach chorale ---
notes_df = read_csv(
    EXAMPLE_DIR / "Bach,_Johann_Sebastian_Chorales_010_notes.csv"
).reset_index()
chords_df = pd.read_csv(
    EXAMPLE_DIR / "Bach,_Johann_Sebastian_Chorales_010_chords.csv",
    dtype={"chord_pcs": str},
)

# Build (onset, label) annotations from the chord df
annotations = [
    (row.onset, f"{row.key}.{row.primary_degree}{row.quality}")
    for _, row in chords_df.iterrows()
]

# Plot first 16 beats as an excerpt
excerpt = notes_df[
    (notes_df.type.isin(["note", "bar"])) & (notes_df.onset < 16)
].reset_index(drop=True)
excerpt_annotations = [(o, l) for o, l in annotations if o < 16]

fig, ax = plt.subplots(figsize=(14, 5))
plot_piano_roll(
    excerpt,
    annotations=excerpt_annotations,
    barlines=True,
    ax=ax,
    title="Bach Chorale 010 — with chord annotations",
)
plt.show()
