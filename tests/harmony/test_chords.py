import pytest

from music_df.harmony.chords import get_rn_pitch_classes

TEST_CASES: list[tuple[dict, list[tuple[str, str, str]]]] = [
    (
        {},
        [
            ("I", "M", "047"),
            ("i", "m", "037"),
            ("I", "m", "047"),
            ("V", "M", "7b2"),
            ("V", "m", "7b2"),
            ("v", "m", "7a2"),
            ("I6", "C", "470"),
            ("I6", "Eb", "7a3"),
        ],
    ),
    (
        {"case_matters": False},
        [
            ("I", "M", "047"),
            ("i", "m", "037"),
            ("I", "m", "037"),
            ("V", "M", "7b2"),
            # NB: if case_matters=False, then the minor dominant is itself minor.
            ("V", "m", "7a2"),
            ("v", "m", "7a2"),
            ("I6", "C", "470"),
            ("I6", "Eb", "7a3"),
        ],
    ),
    (
        {"rn_format": "rnbert"},
        [
            ("IM", "M", "047"),
            ("Im", "m", "037"),
            ("IM", "m", "047"),
            ("VM", "M", "7b2"),
            ("VM", "m", "7b2"),
            ("Vm", "m", "7a2"),
            ("I6M", "C", "470"),
            ("I6M", "Eb", "7a3"),
            # Secondary mode: III of G major vs III of G minor
            ("IIIM/VM", "C", "b36"),
            ("IIIM/Vm", "C", "a25"),
        ],
    ),
]


@pytest.mark.parametrize("kwargs, test_cases", TEST_CASES)
def test_get_rn_pitch_classes_hex_str(kwargs, test_cases):
    for rn, mode_or_key, expected in test_cases:
        result = get_rn_pitch_classes(rn, mode_or_key, hex_str=True, **kwargs)
        assert result == expected, f"{rn}, {mode_or_key}, {expected=}, {result=}"
