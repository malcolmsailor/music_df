from functools import partial
from typing import Optional

import pandas as pd

ALPHABET = "fcgdaeb".upper()


def spelling_to_line_of_fifths(spelling, sharp_char="#", flat_char="b"):
    """Convert from a spelling to a "fifth-class" where F is 0.

    >>> spelling_to_line_of_fifths("C")
    1
    >>> spelling_to_line_of_fifths("C#")
    8
    >>> spelling_to_line_of_fifths("C##")
    15
    >>> spelling_to_line_of_fifths("Fbbb")
    -21
    """
    letter = ALPHABET.index(spelling[0])
    sharps = spelling.count(sharp_char)
    flats = spelling.count(flat_char)
    return letter + 7 * sharps + -7 * flats


def line_of_fifths_to_spelling(i, sharp_char="#", flat_char="b"):
    """Convert from a "fifth-class" (where F is 0) to a spelling.

    >>> line_of_fifths_to_spelling(0)
    'F'
    >>> line_of_fifths_to_spelling(14)
    'F##'
    >>> line_of_fifths_to_spelling(-13)
    'Cbb'
    """
    sharps, letter = divmod(i, 7)
    if sharps >= 0:
        return ALPHABET[letter] + sharp_char * sharps
    return ALPHABET[letter] + flat_char * (-sharps)
