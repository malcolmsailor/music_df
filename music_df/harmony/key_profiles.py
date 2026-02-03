"""
Key profiles for key-finding algorithms.

Each profile is a 12-element tuple where index 0 corresponds to the tonic,
index 1 to the minor 2nd, etc.

References:
- Krumhansl, C. L. (1990). Cognitive Foundations of Musical Pitch.
- Aarden, B. (2003). Dynamic Melodic Expectancy. PhD dissertation, Ohio State University.
- Temperley, D. (2007). Music and Probability. MIT Press. Chapter 6, Table 6.1.
"""

from typing import Sequence

# Krumhansl-Kessler probe-tone ratings (Krumhansl 1990)
KRUMHANSL_MAJOR: tuple[float, ...] = (
    6.39,  # 1 (tonic)
    2.23,  # b2
    3.48,  # 2
    2.32,  # b3
    4.40,  # 3
    4.11,  # 4
    2.50,  # b5
    5.21,  # 5
    2.37,  # b6
    3.66,  # 6
    2.29,  # b7
    2.88,  # 7
)

KRUMHANSL_MINOR: tuple[float, ...] = (
    6.33,  # 1 (tonic)
    2.68,  # b2
    3.51,  # 2
    5.36,  # b3
    2.58,  # 3
    3.50,  # 4
    2.51,  # b5
    4.73,  # 5
    3.96,  # b6
    2.66,  # 6
    3.32,  # b7
    3.14,  # 7
)

# Aarden-Essen corpus-derived counts (Aarden 2003)
AARDEN_MAJOR: tuple[int, ...] = (
    12025,  # 1 (tonic)
    175,    # b2
    10225,  # 2
    200,    # b3
    13400,  # 3
    7725,   # 4
    400,    # b5
    14750,  # 5
    150,    # b6
    5525,   # 6
    325,    # b7
    3575,   # 7
)

AARDEN_MINOR: tuple[int, ...] = (
    523,  # 1 (tonic)
    21,   # b2
    401,  # 2
    481,  # b3
    19,   # 3
    408,  # 4
    20,   # b5
    533,  # 5
    139,  # b6
    73,   # 6
    203,  # b7
    65,   # 7
)

# CBMS profiles (Temperley 2001a, via Temperley 2007 Table 6.1)
CBMS_MAJOR: tuple[float, ...] = (
    5.0,  # 1 (tonic)
    2.0,  # b2
    3.5,  # 2
    2.0,  # b3
    4.5,  # 3
    4.0,  # 4
    2.0,  # b5
    4.5,  # 5
    2.0,  # b6
    3.5,  # 6
    1.5,  # b7
    4.0,  # 7
)

CBMS_MINOR: tuple[float, ...] = (
    5.0,  # 1 (tonic)
    2.0,  # b2
    3.5,  # 2
    4.5,  # b3
    2.0,  # 3
    4.0,  # 4
    2.0,  # b5
    4.5,  # 5
    3.5,  # b6
    2.0,  # 6
    1.5,  # b7
    4.0,  # 7
)

# Kostka-Payne corpus profiles (Temperley 2007 Table 6.1)
# Values represent proportion of segments containing each scale degree
KOSTKA_PAYNE_MAJOR: tuple[float, ...] = (
    0.748,  # 1 (tonic)
    0.060,  # b2
    0.488,  # 2
    0.082,  # b3
    0.670,  # 3
    0.460,  # 4
    0.096,  # b5
    0.715,  # 5
    0.104,  # b6
    0.366,  # 6
    0.057,  # b7
    0.400,  # 7
)

KOSTKA_PAYNE_MINOR: tuple[float, ...] = (
    0.712,  # 1 (tonic)
    0.084,  # b2
    0.474,  # 2
    0.618,  # b3
    0.049,  # 3
    0.460,  # 4
    0.105,  # b5
    0.747,  # 5
    0.404,  # b6
    0.067,  # 6
    0.133,  # b7
    0.330,  # 7
)

# Temperley corpus profiles (Temperley 2007 Table 6.1)
# Values represent proportion of segments containing each scale degree
TEMPERLEY_MAJOR: tuple[float, ...] = (
    0.811,  # 1 (tonic)
    0.024,  # b2
    0.659,  # 2
    0.074,  # b3
    0.721,  # 3
    0.616,  # 4
    0.117,  # b5
    0.835,  # 5
    0.088,  # b6
    0.430,  # 6
    0.031,  # b7
    0.544,  # 7
)

TEMPERLEY_MINOR: tuple[float, ...] = (
    0.786,  # 1 (tonic)
    0.058,  # b2
    0.618,  # 2
    0.734,  # b3
    0.052,  # 3
    0.618,  # 4
    0.185,  # b5
    0.763,  # 5
    0.497,  # b6
    0.104,  # 6
    0.139,  # b7
    0.399,  # 7
)

KEY_PROFILES: dict[str, Sequence[float]] = {
    "krumhansl_major": KRUMHANSL_MAJOR,
    "krumhansl_minor": KRUMHANSL_MINOR,
    "aarden_major": AARDEN_MAJOR,
    "aarden_minor": AARDEN_MINOR,
    "cbms_major": CBMS_MAJOR,
    "cbms_minor": CBMS_MINOR,
    "kostka_payne_major": KOSTKA_PAYNE_MAJOR,
    "kostka_payne_minor": KOSTKA_PAYNE_MINOR,
    "temperley_major": TEMPERLEY_MAJOR,
    "temperley_minor": TEMPERLEY_MINOR,
}
