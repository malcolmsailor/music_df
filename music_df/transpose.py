import functools
from collections import defaultdict
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
    """

    def __init__(self):
        self._memo = defaultdict(dict)

    def __call__(self, spelling, interval):
        if spelling in self._memo[interval]:
            return self._memo[interval][spelling]
        else:
            new_spelling = line_of_fifths_to_spelling(
                spelling_to_line_of_fifths(spelling) + interval
            )
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
    """

    def __init__(self):
        self._memo = defaultdict(dict)

    def __call__(self, midi_num: int, interval: int, pc: bool = False) -> int:
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
    new_key: int,
    inplace: bool = True,
):
    """
    Dataframe must have a "global_key" int attribute in df.attrs. The int
    indicates the number of sharps/flats in the key signature.

    >>> df = pd.DataFrame(
    ...     {"pitch": [60, 62, 64], "spelling": ["C", "D", "E"], "pc": [0, 2, 4]}
    ... )
    >>> df.attrs["global_key"] = 0

    In order for spelled columns and pc columns to be transposed correctly,
    they need to be included in sequences in df.attrs:
    >>> df.attrs["spelled_columns"] = ("spelling",)
    >>> df.attrs["pc_columns"] = ("pc",)

    >>> new_df = transpose_to_key(df, 2, inplace=False)
    >>> new_df
       pitch spelling  pc
    0     62        D   2
    1     64        E   4
    2     66       F#   6
    >>> new_df.attrs["global_key"]
    2
    >>> new_df.attrs["transposed_by_n_sharps"]
    2
    >>> newer_df = transpose_to_key(new_df, -7, inplace=False)
    >>> newer_df
       pitch spelling  pc
    0     59       Cb  11
    1     61       Db   1
    2     63       Eb   3
    >>> newer_df.attrs["global_key"]
    -7
    >>> newer_df.attrs["transposed_by_n_sharps"]
    -7
    """
    orig_key = df.attrs["global_key"]
    transposed_by = df.attrs.get("transposed_by_n_sharps", 0)
    interval = new_key - orig_key
    out_df = df if inplace else df.copy()
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

    out_df.attrs["global_key"] = new_key
    out_df.attrs["transposed_by_n_sharps"] = transposed_by + interval
    return out_df
