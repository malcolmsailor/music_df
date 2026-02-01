import re
import warnings
from typing import Callable, Literal, TypeVar, overload

try:
    import mspell
except ImportError as e:
    raise ImportError(
        "mspell is required for this feature. "
        "Install with: pip install music_df[harmony]"
    ) from e

try:
    from music21.chord import Chord
    from music21.key import Key
    from music21.roman import Minor67Default, RomanNumeral, romanNumeralFromChord
except ImportError as e:
    raise ImportError(
        "music21 is required for this feature. "
        "Install with: pip install music_df[harmony]"
    ) from e

from music_df.keys import MAJOR_KEYS, MINOR_KEYS
from music_df.transpose import SPELLING_TRANSPOSER
from music_df.utils._types import MinorScaleType, Mode
from music_df.utils.music21_caching import KEY_CACHE

K = TypeVar("K")
V = TypeVar("V")

MODE_TO_DEFAULT_KEY: dict[Mode, str] = {"M": "C", "m": "c"}
SPELLER = mspell.Speller()
UNSPELLER = mspell.Unspeller()


def pc_ints_to_hex_str(pcs: list[int]) -> str:
    """
    >>> pc_ints_to_hex_str([0, 4, 7, 11])
    '047b'
    >>> pc_ints_to_hex_str([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    '0123456789ab'
    """
    return "".join(f"{pc:x}" for pc in pcs)


def hex_str_to_pc_ints(hex_str: str, return_set: bool = False) -> list[int] | set[int]:
    """
    >>> hex_str_to_pc_ints("047b")
    [0, 4, 7, 11]
    >>> hex_str_to_pc_ints("0123456789ab")
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """
    if return_set:
        return {int(ch, 16) for ch in hex_str}
    return [int(ch, 16) for ch in hex_str]


class CacheDict(dict[K, V]):
    def __init__(self, default_factory: Callable[[K], V], *args, **kwargs):
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: K) -> V:
        if key not in self:
            self[key] = self.default_factory(key)
        return super().__getitem__(key)


def transpose_pcs(pcs: list[int], interval: int) -> list[int]:
    """
    Transpose a list of pitch classes by an interval.
    """
    return [(pc + interval) % 12 for pc in pcs]


@overload
def get_rn_pitch_classes(
    rn: str,
    mode_or_key: str,
    sixth_minor: Minor67Default = ...,
    seventh_minor: Minor67Default = ...,
    case_matters: bool = ...,
    rn_format: Literal["rnbert", "music21"] = ...,
    rn_translation_cache: CacheDict[str, str] | None = ...,
    *,
    hex_str: Literal[True],
) -> str: ...


@overload
def get_rn_pitch_classes(
    rn: str,
    mode_or_key: str,
    sixth_minor: Minor67Default = ...,
    seventh_minor: Minor67Default = ...,
    case_matters: bool = ...,
    rn_format: Literal["rnbert", "music21"] = ...,
    rn_translation_cache: CacheDict[str, str] | None = ...,
    hex_str: Literal[False] = ...,
) -> list[int]: ...


@overload
def get_rn_pitch_classes(
    rn: str,
    mode_or_key: str,
    sixth_minor: Minor67Default = ...,
    seventh_minor: Minor67Default = ...,
    case_matters: bool = ...,
    rn_format: Literal["rnbert", "music21"] = ...,
    rn_translation_cache: CacheDict[str, str] | None = ...,
    hex_str: bool = ...,
) -> list[int] | str: ...


def get_rn_pitch_classes(
    rn: str,
    mode_or_key: Mode | str,
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
    case_matters: bool = True,
    rn_format: Literal["rnbert", "music21"] = "music21",
    rn_translation_cache: CacheDict[str, str] | None = None,
    hex_str: bool = False,
) -> list[int] | str:
    """
    Get the pitch classes of a Roman numeral chord.

    >>> get_rn_pitch_classes("I", "M")
    [0, 4, 7]

    >>> get_rn_pitch_classes("i", "m")
    [0, 3, 7]

    >>> get_rn_pitch_classes("i", "c")
    [0, 3, 7]

    >>> get_rn_pitch_classes("Im", "c", rn_format="rnbert")
    [0, 3, 7]

    >>> get_rn_pitch_classes("I6", "C")
    [4, 7, 0]

    >>> get_rn_pitch_classes("I", "Eb", hex_str=True)
    '37a'

    >>> get_rn_pitch_classes("Im6", "c", rn_format="rnbert")
    [3, 7, 0]

    """

    key = MODE_TO_DEFAULT_KEY.get(mode_or_key, mode_or_key)  # type:ignore

    if rn_translation_cache is not None:
        rn = rn_translation_cache[rn]
    elif rn_format == "rnbert":
        assert case_matters, (
            "case_matters must be True with rn_format=rnbert. Even though the format only uses upper case, we translate to case-sensitive music21 symbols for processing"
        )
        rn = translate_rns(rn, src="rnbert", dst="music21")

    roman = RomanNumeral(
        rn,
        key,
        sixthMinor=sixth_minor,
        seventhMinor=seventh_minor,
        caseMatters=case_matters,
    )

    pitch_classes = roman.pitchClasses
    if key is not None:
        pitch_classes = transpose_pcs(pitch_classes, KEY_CACHE[key].tonic.pitchClass)
    if hex_str:
        return pc_ints_to_hex_str(pitch_classes)
    return pitch_classes


def get_rn_pc_cache(
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
    case_matters: bool = True,
    hex_str: bool = False,
    rn_format: Literal["rnbert", "music21"] = "music21",
) -> CacheDict[tuple[str, str], list[int] | str]:
    """
    >>> rn_pc_cache = get_rn_pc_cache()
    >>> rn_pc_cache["I", "M"]
    [0, 4, 7]
    >>> rn_pc_cache["I", "M"]
    [0, 4, 7]
    >>> hex_rn_pc_cache = get_rn_pc_cache(hex_str=True)
    >>> hex_rn_pc_cache["I", "Eb"]
    '37a'

    >>> hex_rn_pc_cache["V/V", "C"]
    '269'
    >>> rnbert_cache = get_rn_pc_cache(hex_str=True, rn_format="rnbert")
    >>> rnbert_cache["VM/V", "C"]
    '269'
    >>> rnbert_cache["IVm/V", "C"]
    '037'
    >>> rnbert_cache["IM/bII", "Bb"]
    'b36'


    We can double-check that the caching is working by modifying the cached value
    (which in normal circumstances we should avoid ever doing):
    >>> cached_value = rn_pc_cache["I", "M"]
    >>> cached_value[-1] = "foobar"
    >>> rn_pc_cache["I", "M"]
    [0, 4, 'foobar']
    """
    if rn_format == "rnbert":
        assert case_matters, (
            "case_matters must be True with rn_format=rnbert. Even though the format only uses upper case, we translate to case-sensitive music21 symbols for processing"
        )
        rn_translation_cache = get_rn_translation_cache(src="rnbert", dst="music21")
    else:
        rn_translation_cache = None

    def factory(rn_and_mode_or_key: tuple[str, str]) -> list[int] | str:
        return get_rn_pitch_classes(
            *rn_and_mode_or_key,
            sixth_minor=sixth_minor,
            seventh_minor=seventh_minor,
            case_matters=case_matters,
            hex_str=hex_str,
            rn_format=rn_format,
            rn_translation_cache=rn_translation_cache,
        )

    return CacheDict(factory)


SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "harmonic": [0, 2, 3, 5, 7, 8, 11],
    "natural": [0, 2, 3, 5, 7, 8, 10],
    "melodic": [0, 2, 3, 5, 7, 9, 11],
}


def get_key_pitch_classes(
    key: str, minor_scale_type: MinorScaleType = "harmonic", hex_str: bool = False
) -> list[int] | str:
    """
    Get the pitch classes of a key.

    >>> get_key_pitch_classes("c")
    [0, 2, 3, 5, 7, 8, 11]

    >>> get_key_pitch_classes("C")
    [0, 2, 4, 5, 7, 9, 11]

    >>> get_key_pitch_classes("a", minor_scale_type="natural")
    [9, 11, 0, 2, 4, 5, 7]
    >>> get_key_pitch_classes("a-", minor_scale_type="melodic")
    [8, 10, 11, 1, 3, 5, 7]

    """
    k = Key(key)

    if k.mode == "major":
        out = [(p + k.tonic.pitchClass) % 12 for p in SCALES["major"]]
    else:
        out = [(p + k.tonic.pitchClass) % 12 for p in SCALES[minor_scale_type]]
    if hex_str:
        return pc_ints_to_hex_str(out)
    return out


def get_key_pc_cache(
    minor_scale_type: MinorScaleType = "harmonic", hex_str: bool = False
) -> CacheDict[str, list[int] | str]:
    """
    >>> key_pc_cache = get_key_pc_cache()
    >>> key_pc_cache["c"]
    [0, 2, 3, 5, 7, 8, 11]
    >>> key_pc_cache["c"]
    [0, 2, 3, 5, 7, 8, 11]
    >>> hex_key_pc_cache = get_key_pc_cache(hex_str=True)
    >>> hex_key_pc_cache["c"]
    '023578b'

    We can double-check that the caching is working by modifying the cached value
    (which in normal circumstances we should avoid ever doing):
    >>> cached_value = key_pc_cache["c"]
    >>> cached_value[-1] = "foobar"
    >>> key_pc_cache["c"]
    [0, 2, 3, 5, 7, 8, 'foobar']


    """
    key_pc_cache: CacheDict[str, list[int] | str] = CacheDict(
        lambda key: get_key_pitch_classes(key, minor_scale_type, hex_str)
    )
    return key_pc_cache


def translate_rns(
    rn: str,
    src: Literal["rnbert", "music21"] = "rnbert",
    dst: Literal["rnbert", "music21"] = "music21",
) -> str:
    """
    >>> translate_rns("Im")
    'i'
    >>> translate_rns("VM")
    'V'
    >>> translate_rns("IVm6")
    'iv6'
    >>> translate_rns("VIIo642")
    'viio642'
    >>> translate_rns("III+")
    'III+'

    >>> translate_rns("xaug665")
    'Ger65'
    >>> translate_rns("xaug643")
    'Fr43'

    # TODO: (Malcolm 2026-01-22) improve augmented sixth handling
    All other "inversions" should return It6 for now but this should be improved on!
    >>> translate_rns("xaug63")
    'It6'
    >>> translate_rns("xaug642")
    'It6'

    >>> translate_rns("vM")
    Traceback (most recent call last):
    ...
    TypeError: Mal-formed rnbert Roman numeral begins with lower-case: vM
    """

    if src != "rnbert":
        raise NotImplementedError
    else:
        if dst != "music21":
            raise NotImplementedError

        match rn:
            case "xaug665":
                return "Ger65"
            case "xaug643":
                return "Fr43"
        if rn.startswith("xaug6"):
            return "It6"

        m = re.match(r"[IV]+", rn)
        if m is None:
            if src == "rnbert" and re.match(r"^[iv]", rn):
                raise TypeError(
                    f"Mal-formed rnbert Roman numeral begins with lower-case: {rn}"
                )
            else:
                return rn
        degree = m.group(0)
        match rn[len(degree)]:
            case "M":
                return degree.upper() + rn[len(degree) + 1 :]
            case "m":
                return degree.lower() + rn[len(degree) + 1 :]
            case "o":
                return degree.lower() + rn[len(degree) :]
            case _:
                return rn


def get_rn_translation_cache(
    src: Literal["rnbert", "music21"] = "rnbert",
    dst: Literal["rnbert", "music21"] = "music21",
) -> CacheDict[str, str]:
    """
    >>> cache = get_rn_translation_cache()
    >>> cache["Im"]
    'i'
    >>> cache["VM"]
    'V'
    >>> cache["IVm6"]
    'iv6'
    """
    rn_translation_cache: CacheDict[str, str] = CacheDict(
        lambda rn: translate_rns(rn, src, dst)
    )
    return rn_translation_cache


# mode: {secondary_rn: (fifths_offset, mode)}
TONICIZATIONS = {
    "M": {
        "II": (2, "m"),
        "III": (4, "m"),
        "IV": (-1, "M"),
        "V": (1, "M"),
        "VI": (3, "m"),
        "VII": (5, "m"),
        "bI": (-7, "M"),
        "bII": (-5, "M"),
        "bIII": (-3, "M"),
        "bIV": (-8, "M"),
        "bV": (-6, "M"),
        "bVI": (-4, "M"),
        "bVII": (-2, "M"),
        "#I": (7, "M"),
        "#II": (9, "M"),
        "#III": (11, "M"),
        "#IV": (6, "M"),
        "#V": (8, "M"),
        "#VI": (10, "M"),
        "#VII": (12, "M"),
    },
    "m": {
        "II": (2, "m"),
        "III": (-3, "M"),
        "IV": (-1, "m"),
        "V": (1, "m"),
        "VI": (
            -4,
            "M",
        ),  # TODO: (Malcolm 2025-06-05) double-check, or allow different versions
        "VII": (
            -2,
            "M",
        ),  # TODO: (Malcolm 2025-06-05) double-check, or allow different versions
        "bI": (-7, "M"),
        "bII": (-5, "M"),
        "bIII": (-10, "M"),
        "bIV": (-8, "M"),
        "bV": (-6, "M"),
        "bVI": (
            -11,
            "M",
        ),  # TODO: (Malcolm 2025-06-05) double-check, or allow different versions
        "bVII": (
            -13,
            "M",
        ),  # TODO: (Malcolm 2025-06-05) double-check, or allow different versions
        "#I": (7, "M"),
        "#II": (9, "M"),
        "#III": (4, "M"),
        "#IV": (6, "M"),
        "#V": (8, "M"),
        "#VI": (
            3,
            "m",
        ),  # TODO: (Malcolm 2025-06-05) double-check, or allow different versions
        "#VII": (
            5,
            "m",
        ),  # TODO: (Malcolm 2025-06-05) double-check, or allow different versions
    },
}
# We also need to add lower-case versions of the secondary RNs to the TONICIZATIONS
for mode in TONICIZATIONS.keys():
    values = list(TONICIZATIONS[mode].items())
    for rn, (fifths_offset, secondary_mode) in values:
        # We can't simply do rn.lower() because we need to preserve 'b'
        new_rn = rn.replace("V", "v").replace("I", "i")
        TONICIZATIONS[mode][new_rn] = (fifths_offset, secondary_mode)

MODES = {0: "m", 1: "M"}

# MAJOR_KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# MINOR_KEYS = ["c", "c#", "d", "eb", "e", "f", "f#", "g", "g#", "a", "bb", "b"]


def tonicization_to_key(
    secondary_rn: str,
    original_key: str,
    case_matters: bool = False,
    simplify_enharmonics: bool = True,
) -> str:
    """
    Given the secondary RN and the original key, return the key of the tonicization.

    If the secondary RNs don't use case to indicate mode, then mode is under-specified
    and we follow the following heuristic:
    - if there is a consonant triad in the diatonic scale corresponding to the secondary
        RN (or nearly so in the case of VII in major and II in minor), then we use
        the mode of this triad. Similarly for #VI and #VII in minor which are treated
        as "honorary diatonics".
    - otherwise, we use the major mode. This is questionable in certain cases (e.g.,
        #I in minor, which would for example in C minor be C# major) but these cases
        should be extremely rare in any real data.
    - if simplify_enharmonics is True, return the "enharmonically simplest" equivalent
        key (e.g., "Eb" rather than "D#"). See MAJOR_KEYS and MINOR_KEYS for the
        mapping.

    Args:
        secondary_rn: The secondary Roman numeral.
        original_key: The original key.
        case_matters: Whether case of the secondary RN indicates mode.

    >>> tonicization_to_key("ii", "C")
    'd'

    >>> tonicization_to_key("V", "C")
    'G'
    >>> tonicization_to_key("I", "C")
    'C'
    >>> tonicization_to_key("VII", "C")
    'b'
    >>> tonicization_to_key("VII", "C", case_matters=True)
    'B'
    >>> tonicization_to_key("bVII", "C")
    'Bb'
    >>> tonicization_to_key("bII", "Bb")
    'B'

    >>> tonicization_to_key("III", "C")
    'e'
    >>> tonicization_to_key("III", "c")
    'Eb'
    >>> tonicization_to_key("#III", "C")
    'F'
    >>> tonicization_to_key("#III", "C", simplify_enharmonics=False)
    'E#'


    """
    if secondary_rn == "I":
        return original_key

    mode = MODES[original_key[0].isupper()]
    try:
        fifths_offset, secondary_mode = TONICIZATIONS[mode][secondary_rn]
    except KeyError:
        warnings.warn(f"Unrecognized secondary RN: {secondary_rn}")
        return original_key

    if case_matters:
        secondary_mode = MODES[secondary_rn.lstrip("#b")[0].isupper()]

    if simplify_enharmonics:
        original_key_pc = UNSPELLER(original_key)
        assert isinstance(original_key_pc, int)
        new_key_pc = (original_key_pc + fifths_offset * 7) % 12
        if secondary_mode == "M":
            new_key = MAJOR_KEYS[new_key_pc]
        else:
            new_key = MINOR_KEYS[new_key_pc]
    else:
        new_key = SPELLING_TRANSPOSER(original_key, fifths_offset)

    assert isinstance(new_key, str)

    if secondary_mode == "m":
        new_key = new_key[0].lower() + new_key[1:]
    return new_key


def get_tonicization_cache(
    case_matters: bool = False,
    simplify_enharmonics: bool = True,
) -> CacheDict[tuple[str, str], str]:
    """
    >>> cache = get_tonicization_cache()
    >>> cache[("V", "C")]
    'G'
    """
    tonicization_cache: CacheDict[tuple[str, str], str] = CacheDict(
        lambda secondary_rn_and_original_key: tonicization_to_key(
            *secondary_rn_and_original_key, case_matters, simplify_enharmonics
        )
    )
    return tonicization_cache


def spelled_pitch_to_rn(
    spelled_pitch: str,
    key: str,
    rn_mode: str,
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
):
    """
    Convert a spelled pitch to a Roman numeral in the given key.

    Args:
        spelled_pitch: The spelled pitch.
        key: The key.
        rn_mode: The mode of the Roman numeral.

    >>> spelled_pitch_to_rn("C", "C", "M")
    'I'
    >>> spelled_pitch_to_rn("C#", "C", "M")
    '#I'
    >>> spelled_pitch_to_rn("Eb", "C", "M")
    'bIII'
    >>> spelled_pitch_to_rn("Ab", "C", "M")
    'bVI'
    >>> spelled_pitch_to_rn("A--", "C", "M")
    'bbVI'
    >>> spelled_pitch_to_rn("Ab", "c", "M")
    'VI'
    >>> spelled_pitch_to_rn("A", "C", "M")
    'VI'
    >>> spelled_pitch_to_rn("A", "c", "M")
    '#VI'
    >>> spelled_pitch_to_rn("C", "C", "m")
    'i'
    >>> spelled_pitch_to_rn("C#", "C", "m")
    '#i'
    >>> spelled_pitch_to_rn("Eb", "C", "m")
    'biii'
    >>> spelled_pitch_to_rn("Ab", "C", "m")
    'bvi'
    >>> spelled_pitch_to_rn("A--", "c", "m")
    'bvi'
    >>> spelled_pitch_to_rn("Ab", "c", "m")
    'vi'
    >>> spelled_pitch_to_rn("A", "C", "m")
    'vi'
    >>> spelled_pitch_to_rn("A", "c", "m")
    '#vi'
    """
    # Temporary hack: for some reason this gives VI and VII instead of #VI and #VII *only*
    #   when mode is major but key is minor.

    # Therefore, rather than doing:
    # rn = RomanNumeral("i" if mode == "m" else "I", keyOrScale=pitch.upper())

    # We always do the rn in minor mode then adjust the output as necessary below
    rn = RomanNumeral(
        "i",
        spelled_pitch.upper(),
        sixthMinor=sixth_minor,
        seventhMinor=seventh_minor,
    )

    # (Malcolm 2023-10-05) romanNumeralFromChord always uses the default values
    #   of Minor67Default, meaning it'll sometimes give the wrong answer
    #   for DCML. This is probably few enough cases that it's ok for now.
    out = romanNumeralFromChord(Chord(rn.pitches), key).romanNumeral

    if rn_mode == "M":
        n_flats = len(out) - len(out.lstrip("b"))
        return out[:n_flats] + out[n_flats:].upper()

    return out


def get_spelled_pitch_to_rn_cache(
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
) -> CacheDict[tuple[str, str, str], str]:
    """
    >>> cache = get_spelled_pitch_to_rn_cache()
    >>> cache[("C", "C", "M")]
    'I'
    """
    spelled_pitch_to_rn_cache: CacheDict[tuple[str, str, str], str] = CacheDict(
        lambda spelled_pitch_and_key_and_rn_mode: spelled_pitch_to_rn(
            *spelled_pitch_and_key_and_rn_mode,
            sixth_minor=sixth_minor,
            seventh_minor=seventh_minor,
        )
    )
    return spelled_pitch_to_rn_cache


def rn_to_spelled_pitch(
    rn: str,
    key: str,
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
) -> str:
    """
    >>> rn_to_spelled_pitch("I", "C")
    'C'
    >>> rn_to_spelled_pitch("#I", "C")
    'C#'
    >>> rn_to_spelled_pitch("bIII", "C")
    'E-'

    Note that the mode of both the key and the rn_str can affect the result:
    >>> rn_to_spelled_pitch("VI", "C")
    'A'
    >>> rn_to_spelled_pitch("VI", "c")
    'A-'
    >>> rn_to_spelled_pitch("vi", "c")
    'A'

    Having the mode of the rn_str affect the results can be avoided by setting
    "sixth_minor" and "seventh_minor" appropriately:
    >>> rn_to_spelled_pitch("VI", "c", sixth_minor=Minor67Default.FLAT)
    'A-'
    >>> rn_to_spelled_pitch("vi", "c", sixth_minor=Minor67Default.FLAT)
    'A-'
    """
    out = (
        RomanNumeral(rn, key, sixthMinor=sixth_minor, seventhMinor=seventh_minor)
        .root()
        .name
    )
    return out


def get_rn_to_spelled_pitch_cache(
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
) -> CacheDict[tuple[str, str], str]:
    """
    >>> cache = get_rn_to_spelled_pitch_cache()
    >>> cache[("I", "C")]
    'C'
    """
    rn_to_spelled_pitch_cache: CacheDict[tuple[str, str], str] = CacheDict(
        lambda rn_and_key: rn_to_spelled_pitch(
            *rn_and_key, sixth_minor=sixth_minor, seventh_minor=seventh_minor
        )
    )
    return rn_to_spelled_pitch_cache


def match_pitch_case_to_rn_mode(rn: str, pitch: str) -> str:
    """
    >>> match_pitch_case_to_rn_mode("iii", "E")
    'e'
    >>> match_pitch_case_to_rn_mode("III", "E")
    'E'
    >>> match_pitch_case_to_rn_mode("biii", "E-")
    'e-'
    >>> match_pitch_case_to_rn_mode("bIII", "E-")
    'E-'
    >>> match_pitch_case_to_rn_mode("#VI", "A#")
    'A#'
    >>> match_pitch_case_to_rn_mode("#vi", "A#")
    'a#'
    """
    assert rn[-1] in {"i", "I", "v", "V"}
    return pitch.capitalize() if rn[-1].isupper() else pitch.lower()


def get_mode_from_rn_str(rn: str) -> Mode:
    assert rn[-1] in {"i", "I", "v", "V"}
    return "M" if rn[-1].isupper() else "m"


def change_rn_mode(rn: str, mode: Mode) -> str:
    n_flats = len(rn) - len(rn.lstrip("b"))
    if mode == "M":
        return rn[:n_flats] + rn[n_flats:].upper()
    return rn[:n_flats] + rn[n_flats:].lower()


def handle_nested_secondary_rns(
    token: str,
    mode_or_key: Mode | str,
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
    rn_to_spelled_pitch_cache: CacheDict[tuple[str, str], str] | None = None,
) -> str:
    """
    >>> handle_nested_secondary_rns("vi/III", "M")
    '#i'
    >>> handle_nested_secondary_rns("vi/III", "Ab")
    '#i'
    >>> handle_nested_secondary_rns("V/V", "M")
    'II'
    >>> handle_nested_secondary_rns("/V/V", "M")  # leading slash is optional
    'II'
    >>> handle_nested_secondary_rns("V/V/V", "M")
    'VI'
    >>> handle_nested_secondary_rns("V/V/V/V", "M")
    'III'
    >>> handle_nested_secondary_rns("v/V", "M")
    'ii'
    >>> handle_nested_secondary_rns("vi/III", "m")
    'i'
    >>> handle_nested_secondary_rns("V", "M")
    'V'
    >>> handle_nested_secondary_rns("", "M")
    'I'
    """
    token = token.lstrip("/")

    if not token:
        return "I"

    if rn_to_spelled_pitch_cache is None:
        get_pitch_from_rn = lambda rn, key: rn_to_spelled_pitch(
            rn, key, sixth_minor, seventh_minor
        )
    else:
        get_pitch_from_rn = lambda rn, key: rn_to_spelled_pitch_cache[(rn, key)]

    dummy_key = {"M": "C", "m": "c"}.get(mode_or_key, mode_or_key)

    # dummy_key = "C" if mode == "M" else "c"

    while token.count("/"):
        bits = token.rsplit("/", maxsplit=2)
        if len(bits) == 3:
            prefix, secondary2, secondary1 = bits
        else:
            prefix = None
            secondary2, secondary1 = bits

        secondary_pitch1 = get_pitch_from_rn(
            secondary1,
            dummy_key,
        )
        secondary_pitch2 = get_pitch_from_rn(
            secondary2,
            match_pitch_case_to_rn_mode(secondary1, secondary_pitch1),
        )

        combined_secondary_without_mode = spelled_pitch_to_rn(
            secondary_pitch2,
            dummy_key,
            get_mode_from_rn_str(secondary2),
            sixth_minor,
            seventh_minor,
        )
        combined_secondary = change_rn_mode(
            combined_secondary_without_mode, get_mode_from_rn_str(secondary2)
        )

        if prefix:
            token = f"{prefix}/{combined_secondary}"
        else:
            token = combined_secondary

    return token


def get_handle_nested_secondary_rns_cache(
    sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
    seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
    rn_to_spelled_pitch_cache: CacheDict[tuple[str, str], str] | None = None,
) -> CacheDict[tuple[str, str], str]:
    """
    >>> cache = get_handle_nested_secondary_rns_cache()
    >>> cache[("vi/III", "M")]
    '#i'
    >>> cache[("vi/III", "B")]
    '#i'
    """
    return CacheDict(
        lambda token_and_mode_or_key: handle_nested_secondary_rns(
            *token_and_mode_or_key,
            sixth_minor=sixth_minor,
            seventh_minor=seventh_minor,
            rn_to_spelled_pitch_cache=rn_to_spelled_pitch_cache,
        )
    )
