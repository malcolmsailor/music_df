# from music21.chord import Chord
# from music21.key import Key
# from music21.pitch import Pitch
# from music21.roman import Minor67Default, RomanNumeral, romanNumeralFromChord

# RN_CACHE: dict[tuple[str, str, Minor67Default, Minor67Default], RomanNumeral] = {}


# def get_rn_from_cache(
#     rn: str,
#     key: str,
#     sixth_minor: Minor67Default = Minor67Default.CAUTIONARY,
#     seventh_minor: Minor67Default = Minor67Default.CAUTIONARY,
# ):
#     if (rn, key, sixth_minor, seventh_minor) in RN_CACHE:
#         return RN_CACHE[(rn, key, sixth_minor, seventh_minor)]
#     roman = RomanNumeral(rn, key, sixthMinor=sixth_minor, seventhMinor=seventh_minor)
#     RN_CACHE[(rn, key, sixth_minor, seventh_minor)] = roman
#     return roman


# RN_FROM_CHORD_CACHE: dict[tuple[tuple[Pitch, ...], str], RomanNumeral] = {}


# def get_rn_from_cache_from_chord(pitches: tuple[Pitch, ...], key: str):
#     if (pitches, key) in RN_FROM_CHORD_CACHE:
#         return RN_FROM_CHORD_CACHE[(pitches, key)]
#     roman = romanNumeralFromChord(Chord(pitches), key)
#     RN_FROM_CHORD_CACHE[(pitches, key)] = roman
#     return roman


# KEY_CACHE: dict[str, Key] = {}


# def get_key_from_cache(key: str) -> Key:
#     if key in KEY_CACHE:
#         return KEY_CACHE[key]
#     key_inst = Key(key)
#     KEY_CACHE[key] = key_inst
#     return key_inst
