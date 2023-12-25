import pandas as pd

from music_df.read_csv import read_csv
from music_df.read_krn import read_krn
from music_df.read_midi import read_midi
from music_df.read_rntxt import read_rntxt
from music_df.read_xml import read_xml


def read(input_path: str, **kwargs) -> pd.DataFrame:
    if input_path.endswith(".mid"):
        return read_midi(input_path, **kwargs)
    elif input_path.endswith(".krn"):
        return read_krn(input_path, sort=True, **kwargs)
    elif input_path.endswith(".csv"):
        out = read_csv(input_path, **kwargs)
        assert out is not None
        return out
    elif any(input_path.endswith(suffix) for suffix in ("xml", "mxl", "mscx", "mscz")):
        return read_xml(input_path, sort=True)
    elif any(input_path.endswith(suffix) for suffix in (".txt", ".rntxt")):
        # Roman text
        return read_rntxt(input_path)
    raise ValueError
