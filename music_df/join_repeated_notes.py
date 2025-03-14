"""
Not yet implemented.
"""

import pandas as pd


def join_repeated_notes(
    df: pd.DataFrame,
    quantize: bool = True,
    ticks_per_quarter: int = 16,
) -> pd.DataFrame:
    if quantize:
        df["quantized_onset"] = (df["onset"] * ticks_per_quarter).round()
        df["quantized_release"] = (df["release"] * ticks_per_quarter).round()
        onset_column = "quantized_onset"
        release_column = "quantized_release"
    else:
        onset_column = "onset"
        release_column = "release"
    raise NotImplementedError
