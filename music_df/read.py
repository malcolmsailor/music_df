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

import pandas as pd

from music_df.read_csv import read_csv
from music_df.read_krn import read_krn
from music_df.read_midi import read_midi
from music_df.read_rntxt import read_rntxt
from music_df.read_xml import read_xml


def read(input_path: str, **kwargs) -> pd.DataFrame:
    if any(input_path.lower().endswith(suffix) for suffix in (".mid", ".midi")):
        return read_midi(input_path, **kwargs)
    elif input_path.lower().endswith(".krn"):
        return read_krn(input_path, sort=True, **kwargs)
    elif input_path.lower().endswith(".csv"):
        out = read_csv(input_path, **kwargs)
        assert out is not None
        return out
    elif any(
        input_path.lower().endswith(suffix) for suffix in ("xml", "mxl", "mscx", "mscz")
    ):
        return read_xml(input_path, sort=True)
    elif any(input_path.lower().endswith(suffix) for suffix in (".txt", ".rntxt")):
        # Roman text
        return read_rntxt(input_path)
    raise ValueError(f"Unsupported file extension: {input_path}")
