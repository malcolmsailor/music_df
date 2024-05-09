import pandas as pd
import io
from music_df.utils.pitch_classes import (
    get_figured_bass_class,
    get_pitch_classes_as_str,
)
import traceback, pdb, sys


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


def add_figured_bass_class(df: pd.DataFrame):
    """
    Groups by onset, which means if the data isn't salami-sliced the result will
    be incorrect.
    >>> csv_table = '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,2.0
    ... note,71,0.0,2.0
    ... note,59,2.0,4.0
    ... note,62,2.0,4.0
    ... note,67,2.0,4.0
    ... note,74,2.0,4.0
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df
       type  pitch  onset  release
    0   bar    NaN    0.0      4.0
    1  note   60.0    0.0      2.0
    2  note   71.0    0.0      2.0
    3  note   59.0    2.0      4.0
    4  note   62.0    2.0      4.0
    5  note   67.0    2.0      4.0
    6  note   74.0    2.0      4.0
    >>> add_figured_bass_class(df)  # doctest: +NORMALIZE_WHITESPACE
       type  pitch  onset  release fb_class
    0   bar    NaN    0.0      4.0
    1  note   60.0    0.0      2.0       0b
    2  note   71.0    0.0      2.0       0b
    3  note   59.0    2.0      4.0      038
    4  note   62.0    2.0      4.0      038
    5  note   67.0    2.0      4.0      038
    6  note   74.0    2.0      4.0      038
    """
    df["fb_class"] = ""
    for onset, group in df[df["type"] == "note"].groupby("onset"):
        figured_bass_class = get_figured_bass_class(group["pitch"].tolist())
        str_repr = get_pitch_classes_as_str(figured_bass_class)
        df.loc[group.index, "fb_class"] = str_repr
    return df
