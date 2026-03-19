"""
Provides a function, `read()`, for reading a file into a music_df.

The allowed extensions are:

- .mid or .midi for MIDI files
- .krn for Humdrum .krn files. Note however that this won't work until I
    distribute my `totable` command-line tool.
- .csv for CSV files
- .xml, .mxl, .mscx, or .mscz for encoded MusicXML files
- .txt or .rntxt for Roman numeral text files
"""

from __future__ import annotations

import pandas as pd

from music_df.read_csv import read_csv
from music_df.read_krn import read_krn
from music_df.read_midi import read_midi
from music_df.read_rntxt import read_rntxt
from music_df.read_xml import read_xml
from music_df.sort_df import sort_df
from music_df.transforms import apply_transforms


def read(
    input_path: str,
    *,
    transforms: list[dict[str, dict]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    lower = input_path.lower()
    if any(lower.endswith(suffix) for suffix in (".mid", ".midi")):
        df = read_midi(input_path, **kwargs)
    elif lower.endswith(".krn"):
        df = read_krn(input_path, sort=True, **kwargs)
    elif lower.endswith(".csv"):
        df = read_csv(input_path, **kwargs)
        assert df is not None
    elif any(lower.endswith(suffix) for suffix in ("xml", "mxl", "mscx", "mscz")):
        df = read_xml(input_path, sort=True)
    elif any(lower.endswith(suffix) for suffix in (".txt", ".rntxt")):
        df = read_rntxt(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {input_path}")

    if transforms is not None:
        df = sort_df(df)
        df = apply_transforms(df, transforms)
    return df
