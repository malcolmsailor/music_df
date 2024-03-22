import functools
from collections import defaultdict
from math import isnan
from typing import Iterable

import pandas as pd

from music_df.utils.spelling import (
    line_of_fifths_to_spelling,
    spelling_to_line_of_fifths,
)

STANDARD_KEYS = list(range(-6, 7))
ENHARMONICALLY_UNIQUE_KEYS = list(range(-6, 6))

SPELLING_MEMO = defaultdict(dict)
MIDI_NUM_MEMO = defaultdict(dict)

ALPHABET = "fcgdaeb".upper()

KEY_CACHE = {}


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


class SpellingAlongLineOfFifthsTransposer:
    """
    >>> transposer = SpellingAlongLineOfFifthsTransposer()
    >>> transposer("C", 2)
    'D'
    >>> transposer("C", 7)
    'C#'
    >>> transposer("C", -9)
    'Bbb'

    We preserve case so that this can also be used to transpose minor keys:
    >>> transposer("c", 3)
    'a'
    >>> transposer("c", -3)
    'eb'

    Nan values are returned unchanged (this is likely to occur when transposing an
    entire column of a DataFrame, where e.g., a key signature does not have a pitch):
    >>> transposer(float("nan"), -3)
    nan

    Likewise, "na" values are returned unchanged for a similar reason:
    >>> transposer("na", -3)
    'na'

    """

    def __init__(self):
        self._memo = defaultdict(dict)

    def __call__(self, spelling: str | float, interval: int) -> str | float:
        if not isinstance(spelling, str) and isnan(spelling):
            return spelling
        if spelling == "na":
            return spelling
        if spelling in self._memo[interval]:
            return self._memo[interval][spelling]
        else:
            assert isinstance(spelling, str)
            new_spelling = line_of_fifths_to_spelling(
                spelling_to_line_of_fifths(spelling.capitalize()) + interval
            )
            if spelling[0].islower():
                new_spelling = new_spelling[0].lower() + new_spelling[1:]
            self._memo[interval][spelling] = new_spelling
            return new_spelling


class MidiNumAlongLineOfFifthsTransposer:
    """
    >>> transposer = MidiNumAlongLineOfFifthsTransposer()
    >>> transposer(60, 2)
    62
    >>> transposer(60, 7)
    61

    Tranposition is always to the nearest pitch:
    >>> transposer(60, 12 * 7)
    60

    We take +6 semitones rather than -6 semitones:
    >>> transposer(60, 6)
    66
    >>> transposer(60, -6)
    66

    >>> transposer(60, -6, pc=True)
    6

    Nan values are returned unchanged (this is likely to occur when transposing an
    entire column of a DataFrame, where e.g., a key signature does not have a pitch):
    >>> transposer(float("nan"), -3)
    nan
    """

    def __init__(self):
        self._memo = defaultdict(dict)

    def __call__(
        self, midi_num: int | float, interval: int, pc: bool = False
    ) -> int | float:
        if isnan(midi_num):
            return midi_num
        if midi_num in self._memo[interval]:
            out = self._memo[interval][midi_num]
        else:
            chromatic_int = (interval * 7) % 12
            if chromatic_int > 6:
                chromatic_int -= 12
            new_midi_num = midi_num + chromatic_int
            self._memo[interval][midi_num] = new_midi_num
            out = new_midi_num
        if pc:
            return out % 12
        return out


# We create global instances so we can share the _memo between calls of
#   transpose_to_key; however this probably isn't ideal.
SPELLING_TRANSPOSER = SpellingAlongLineOfFifthsTransposer()
MIDI_NUM_TRANSPOSER = MidiNumAlongLineOfFifthsTransposer()


def transpose_to_key(
    df: pd.DataFrame,
    new_key_sig: int,
    inplace: bool = True,
):
    """
    Dataframe must have a "global_key_sig" int attribute in df.attrs.
    This attribute indicates the number of sharps/flats in the key signature.

    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 62, 64],
    ...         "spelling": ["C", "D", "E"],
    ...         "pc": [0, 2, 4],
    ...         "key": ["C", "C", "a"],
    ...     }
    ... )
    >>> df.attrs["global_key_sig"] = 0

    In order for spelled columns and pc columns to be transposed correctly,
    they need to be included in sequences in df.attrs:
    >>> df.attrs["spelled_columns"] = ("spelling", "key")
    >>> df.attrs["pc_columns"] = ("pc",)

    >>> new_df = transpose_to_key(df, 2, inplace=False)
    >>> new_df
       pitch spelling  pc key
    0     62        D   2   D
    1     64        E   4   D
    2     66       F#   6   b
    >>> new_df.attrs["global_key_sig"]
    2
    >>> new_df.attrs["transposed_by_n_sharps"]
    2
    >>> newer_df = transpose_to_key(new_df, -7, inplace=False)
    >>> newer_df
       pitch spelling  pc key
    0     59       Cb  11  Cb
    1     61       Db   1  Cb
    2     63       Eb   3  ab
    >>> newer_df.attrs["global_key_sig"]
    -7
    >>> newer_df.attrs["transposed_by_n_sharps"]
    -7
    """
    orig_key = df.attrs["global_key_sig"]
    transposed_by = df.attrs.get("transposed_by_n_sharps", 0)
    interval = new_key_sig - orig_key
    out_df = df if inplace else df.copy()

    if not interval:
        return out_df

    for column in df.attrs.get("pitch_columns", ("pitch",)):
        out_df[column] = df[column].apply(
            functools.partial(MIDI_NUM_TRANSPOSER, interval=interval)
        )
    for column in df.attrs.get("spelled_columns", ()):
        out_df[column] = df[column].apply(
            functools.partial(SPELLING_TRANSPOSER, interval=interval)
        )
    for column in df.attrs.get("pc_columns", ()):
        out_df[column] = df[column].apply(
            functools.partial(MIDI_NUM_TRANSPOSER, interval=interval, pc=True)
        )
    for column in df.attrs.get("key_sig_columns", ()):
        df[column] = df[column] + interval
    for column in df.attrs.get("key_sig_class_columns", ()):
        df[column] = (df[column] + interval) % 12
        df.loc[df[column] > 6, column] -= 12

    out_df.attrs.pop("global_key", None)  # global_key will no longer be accurate
    out_df.attrs["global_key_sig"] = new_key_sig
    out_df.attrs["transposed_by_n_sharps"] = transposed_by + interval
    return out_df
