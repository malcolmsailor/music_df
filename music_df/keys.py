"""
Provides functions for working with keys in music dataframes.
"""

import pandas as pd

try:
    import mspell
except ImportError:
    mspell = None  # type: ignore

ALPHABET = {letter: (i - 1) for (i, letter) in enumerate("FCGDAEB")} | {
    letter: (i - 4) for (i, letter) in enumerate("fcgdaeb")
}

MAJOR_KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
MINOR_KEYS = ("c", "c#", "d", "eb", "e", "f", "f#", "g", "g#", "a", "bb", "b")
UNSPELLER = mspell.Unspeller() if mspell else None


def get_mode(key):
    return "M" if key[0].isupper() else "m"


def simplify_enharmonic_key(key):
    """
    >>> simplify_enharmonic_key("C")
    'C'
    >>> simplify_enharmonic_key("F#")
    'Gb'
    >>> simplify_enharmonic_key("ab")
    'g#'
    """
    if UNSPELLER is None:
        raise ImportError(
            "mspell is required for this feature. "
            "Install with: pip install music_df[humdrum]"
        )
    original_key_pc = UNSPELLER(key)
    assert isinstance(original_key_pc, int)

    if key[0].isupper():
        return MAJOR_KEYS[original_key_pc]
    else:
        return MINOR_KEYS[original_key_pc]


def pc_and_mode_to_key(pc_and_mode: str):
    """
    >>> pc_and_mode_to_key("0.0M")
    'C'
    >>> pc_and_mode_to_key("0.0m")
    'c'
    >>> pc_and_mode_to_key("6.0M")
    'Gb'
    >>> pc_and_mode_to_key("6.0m")
    'f#'
    """
    pc, mode = pc_and_mode[:-1], pc_and_mode[-1]
    pc = int(float(pc))
    if mode == "M":
        return MAJOR_KEYS[pc]
    return MINOR_KEYS[pc]


def key_to_pc_and_mode(key: str):
    """
    >>> key_to_pc_and_mode("D")
    '2.0M'
    >>> key_to_pc_and_mode("Dbb")
    '0.0M'
    >>> key_to_pc_and_mode("bb")
    '10.0m'
    >>> key_to_pc_and_mode("c###")
    '3.0m'
    """
    mode = get_mode(key)

    # far from the most efficient way of doing this
    letter_i = "CDEFGAB".index(key[0].upper())
    pc = [0, 2, 4, 5, 7, 9, 11][letter_i]

    sharps = key[1:].count("#")
    pc = (pc + sharps) % 12
    flats = key[1:].count("b")
    pc = (pc - flats) % 12
    return f"{float(pc)}{mode}"


def key_to_sharps(key_str, minor_offset: float = 0.0):
    """
    >>> key_to_sharps("C")
    0
    >>> key_to_sharps("Cb")
    -7
    >>> key_to_sharps("cb")
    -10
    >>> key_to_sharps("b")
    2
    >>> key_to_sharps("b#")
    9
    >>> key_to_sharps("b", minor_offset=-0.5)
    1.5
    >>> key_to_sharps("Db", minor_offset=-0.5)
    -5
    """
    base = ALPHABET[key_str[0]]

    if minor_offset and key_str[0].islower():
        base += minor_offset

    accs = key_str[1:]
    if not accs:
        return base
    elif accs == "#" * len(accs):
        return base + len(accs) * 7
    elif accs == "b" * len(accs):
        return base - len(accs) * 7
    else:
        raise ValueError(f"Don't understand key in {key_str}")


CHROMATIC_SCALE = (
    {p: i for (i, p) in enumerate("C C# D D# E F F# G G# A A# B".split())}
    | {p: i for (i, p) in enumerate("C Db D Eb E F Gb G Ab A Bb B".split())}
    | {
        p: i
        for (i, p) in enumerate(
            "Dbb Ebbb Ebb Fbb Fb Gbb Abbb Abb Bbbb Bbb Cbb Cb".split()
        )
    }
    | {
        p: i
        for (i, p) in enumerate(
            "B# B## C## C### D## E# E## F## F### G## G### A##".split()
        )
    }
)


def get_key_pc_interval(key1: str, key2: str):
    """
    Returns the interval between the two keys in semitones.

    >>> get_key_pc_interval("C", "C")
    0
    >>> get_key_pc_interval("C", "c")
    0
    >>> get_key_pc_interval("C", "a")
    -3
    >>> get_key_pc_interval("C", "F#")
    -6
    >>> get_key_pc_interval("f#", "C")
    -6
    """
    pc1 = CHROMATIC_SCALE[key1.capitalize()]
    pc2 = CHROMATIC_SCALE[key2.capitalize()]
    pc_interval = (pc2 - pc1) % 12

    if pc_interval >= 6:
        pc_interval -= 12
    return pc_interval


def get_key_sharps_interval(key1: str, key2: str):
    """
    Return the circle-of-fifths interval between the two keys.

    >>> get_key_sharps_interval("C", "C")
    0
    >>> get_key_sharps_interval("C", "c")
    -3
    >>> get_key_sharps_interval("C", "a")
    0
    >>> get_key_sharps_interval("C", "F#")
    -6
    >>> get_key_sharps_interval("C", "Gb")
    -6
    """
    sharps1 = key_to_sharps(key1)
    sharps2 = key_to_sharps(key2)

    sharps_interval = (sharps2 - sharps1) % 12

    if sharps_interval >= 6:
        sharps_interval -= 12

    return sharps_interval


def get_key_interval(key1: str, key2: str):
    """Returns a 2-tuple of the chromatic root interval and the circle of fifths interval.
    >>> get_key_interval("C", "C")
    (0, 0)
    >>> get_key_interval("C", "c")
    (0, -3)
    >>> get_key_interval("C", "a")
    (-3, 0)
    >>> get_key_interval("C", "F#")
    (-6, -6)
    >>> get_key_interval("F#", "C")
    (-6, -6)
    """
    return get_key_pc_interval(key1, key2), get_key_sharps_interval(key1, key2)


def get_key_change_id(key1: str, key2: str):
    """
    Return a "key change id" for the two keys.

    A key change id is a string formed of the concatenation of:
    - the chromatic interval between the two keys,
    - the mode of the first key,
    - the mode of the second key.

    See the examples below.

    >>> get_key_change_id("C", "C")
    '0MM'
    >>> get_key_change_id("C", "c")
    '0Mm'
    >>> get_key_change_id("C", "a")
    '-3Mm'
    >>> get_key_change_id("C", "F#")
    '-6MM'
    >>> get_key_change_id("F#", "C")
    '-6MM'
    """
    pc_interval = get_key_pc_interval(key1, key2)
    mode1 = "M" if key1[0].isupper() else "m"
    mode2 = "M" if key2[0].isupper() else "m"
    return f"{pc_interval}{mode1}{mode2}"


def keys_to_key_change_ints(key_series: pd.Series) -> tuple[dict, pd.Series]:
    """
    Return a dictionary and mask indicating key changes in a series of keys.py

    The dictionary contains the following:
        - key_pc_ints: the chromatic interval between each key and the next.
        - key_sharps_ints: the circle-of-fifths interval between each key and the next.
        - rel_key_pc_ints: the chromatic interval between each key and the global key.
        - rel_key_sharps_ints: the circle-of-fifths interval between each key and the
          global key.

    >>> change_ints, mask = keys_to_key_change_ints(
    ...     pd.Series(["C", "", "F", "f", "ab", "C"])
    ... )
    >>> change_ints  # doctest: +NORMALIZE_WHITESPACE
    {'key_pc_ints': [0, 5, 0, 3, 4],
     'key_sharps_ints': [0, -1, -3, -3, -5],
     'rel_key_pc_ints': [0, 5, 5, -4, 0],
     'rel_key_sharps_ints': [0, -1, -4, 5, 0]}
    >>> mask
    0     True
    1    False
    2     True
    3     True
    4     True
    5     True
    dtype: bool
    """
    key_mask = (~key_series.isna()) & ~(key_series == "")
    keys = key_series[key_mask]

    global_key = keys.iloc[0]
    pc_intervals, sharps_intervals = [0], [0]
    global_pc_intervals, global_sharps_intervals = [0], [0]

    for key1, key2 in zip(keys.iloc[:-1], keys.iloc[1:]):
        pc_interval, sharps_interval = get_key_interval(key1, key2)
        pc_intervals.append(pc_interval)

        sharps_intervals.append(sharps_interval)  # type:ignore

        global_pc_interval, global_sharps_interval = get_key_interval(global_key, key2)
        global_pc_intervals.append(global_pc_interval)
        global_sharps_intervals.append(global_sharps_interval)  # type:ignore

    out = {
        "key_pc_ints": pc_intervals,
        "key_sharps_ints": sharps_intervals,
        "rel_key_pc_ints": global_pc_intervals,
        "rel_key_sharps_ints": global_sharps_intervals,
    }
    return out, key_mask


def keys_to_key_changes(key_series: pd.Series) -> tuple[list[str], pd.Series]:
    """
    Convert a series of keys into a series of key change ids.
    >>> changes, mask = keys_to_key_changes(pd.Series(["C", "", "F", "f", "ab", "C"]))
    >>> changes
    ['C', '5MM', '0Mm', '3mm', '4mM']
    >>> mask
    0     True
    1    False
    2     True
    3     True
    4     True
    5     True
    dtype: bool
    """
    key_mask = (~key_series.isna()) & ~(key_series == "")
    keys = key_series[key_mask]
    global_key = keys.iloc[0]
    key_changes = [global_key]
    for key1, key2 in zip(keys.iloc[:-1], keys.iloc[1:]):
        key_changes.append(get_key_change_id(key1, key2))
    return key_changes, key_mask
