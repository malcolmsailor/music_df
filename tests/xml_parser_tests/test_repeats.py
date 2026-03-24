import math
from fractions import Fraction

from music_df.xml_parser.repeats import get_repeat_segments


def test_endings_without_backward_repeat_treated_as_noop():
    """Endings without any backward repeat barline are an encoding error;
    get_repeat_segments should return the trivial no-repeat result rather
    than producing inf onsets.
    """
    repeats = {
        20: {"start-ending": {"number": "1, 2"}},
        21: {"start-ending": {"number": "3"}},
    }
    measure_ends = {i + 1: Fraction(4 * (i + 1)) for i in range(21)}

    orig, repeated, types = get_repeat_segments(repeats, measure_ends)

    # Should be the trivial no-repeat result
    assert orig == [(0, float("inf"))]
    assert repeated == [(0, float("inf"))]
    assert types == ["no_repeat"]
