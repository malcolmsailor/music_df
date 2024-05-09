from functools import lru_cache
from typing import Sequence


Pitch = int
PitchClass = int


def get_figured_bass_class(pitches: Sequence[Pitch]) -> list[Pitch]:
    """
    >>> get_figured_bass_class([60, 64, 67])
    [0, 4, 7]
    >>> get_figured_bass_class([60, 64, 67, 72])
    [0, 4, 7]
    >>> get_figured_bass_class([62, 66, 69, 74])
    [0, 4, 7]
    """
    return sorted(set(int((p - pitches[0]) % 12) for p in pitches))


def get_pitch_classes_as_str(pcs: Sequence[PitchClass]) -> str:
    """
    >>> get_pitch_classes_as_str([0, 4, 7])
    '047'
    >>> get_pitch_classes_as_str([0, 4, 7, 11])
    '047b'
    >>> get_pitch_classes_as_str([11, 11, 5, 0, 10])
    'bb50a'
    """
    return "".join(hex(p)[2:] for p in pcs)


@lru_cache(maxsize=512)
def get_prime_form(pc_str: str, inversional_equivalence: bool = False) -> str:
    """
    We put pitches in "reverse lexicographic" order.

    Not a fast implementation, but results are small, so we use lru_cache so we can
    quickly apply to large pandas series etc.

    >>> get_prime_form("047")
    '047'
    >>> get_prime_form("038")
    '047'
    >>> get_prime_form("0ab")
    '012'
    >>> get_prime_form("04689")
    '02458'
    >>> get_prime_form("02478")
    '02478'
    >>> get_prime_form("0348a")
    '02478'
    """
    if inversional_equivalence:
        raise NotImplementedError
    input_form = [int("0x" + x, 16) for x in pc_str]
    all_forms = [input_form[i:] + input_form[:i] for i in range(len(input_form))]
    all_forms = [[(x - form[0]) % 12 for x in form][::-1] for form in all_forms]
    sorted_forms = sorted(all_forms)
    return get_pitch_classes_as_str(sorted_forms[0][::-1])
