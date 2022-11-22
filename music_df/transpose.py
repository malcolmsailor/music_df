from collections import defaultdict

import pandas as pd

STANDARD_KEYS = list(range(-6, 7))
ENHARMONICALLY_UNIQUE_KEYS = list(range(-6, 6))

SPELLING_MEMO = defaultdict(dict)
MIDI_NUM_MEMO = defaultdict(dict)

ALPHABET = "fcgdaeb".upper()


def chromatic_transpose(
    df: pd.DataFrame,
    interval: int,
    inplace: bool = True,
    label: bool = False,
    metadata=True,
):
    out_df = df if inplace else df.copy()
    out_df.pitch += interval
    if metadata:
        if "chromatic_transpose" in out_df.attrs:
            out_df.attrs["chromatic_transpose"] += interval
        else:
            out_df.attrs["chromatic_transpose"] = interval
    if label:
        out_df.loc[:, "transposed_by_n_semitones"] = interval
    return out_df
