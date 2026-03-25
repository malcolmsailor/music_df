import re
import typing as t
from collections import defaultdict

try:
    from mspell import Unspeller
except ImportError as e:
    raise ImportError(
        "mspell is required for humdrum export features. "
        "Install with: pip install music_df[humdrum]"
    ) from e

from music_df.humdrum_export.constants import USER_SIGNIFIERS

UNSPELLER = Unspeller(pitches=True, letter_format="kern")


_PITCH_LETTERS = set("abcdefgABCDEFG")


def _is_note_token(token: str):
    return (
        token[0] not in ("*", "!")
        and token[-1] not in (".", "r")
        and any(c in _PITCH_LETTERS for c in token)
    )


TOKEN_PITCH_PATTERN = re.compile(
    r"^.*?(?P<pitch>[a-gA-G]+[#-]*)(?:_|])?(?:"
    + "|".join([re.escape(ch) for ch in (USER_SIGNIFIERS)])
    + r")?$"
)


def _get_token_pitch(token: str):
    """
    >>> _get_token_pitch("16g")
    'g'
    >>> _get_token_pitch("2A--")
    'A--'
    >>> _get_token_pitch("[3.F#")
    'F#'
    """
    m = re.match(TOKEN_PITCH_PATTERN, token)
    assert m is not None, f"Could not extract pitch from token {token!r}"
    return m.group("pitch")


def _sort_measure_spines(measure: t.List[t.List[str]]):
    """
    Takes a list of lists, where each list is one spine of a measure. Sorts
    them from highest to lowest according to their mean pitch.
    """

    def _get_mean_pitch(spine):
        # Chord tokens contain space-separated sub-tokens (e.g., "4FF 4E-")
        sub_tokens = [
            st for t in spine if _is_note_token(t) for st in t.split(" ")
        ]
        pitches = UNSPELLER([_get_token_pitch(st) for st in sub_tokens])
        if not pitches:
            return 0
        return sum(pitches) / len(pitches)

    measure.sort(key=_get_mean_pitch, reverse=True)


def merge_spines(humdrum_contents: str) -> t.List[str]:
    """
    Args:
        humdrum_contents: should consist of humdrum-formatted spines,
            where each spine begins with a **kern token. E.g.,
        TODO example
    """

    def _init_measure():
        nonlocal measure
        nonlocal not_rests
        out = []
        if not any(not_rests):
            not_rests[0] = True
        if measure:
            for i, not_rest in enumerate(not_rests):
                if not_rest:
                    out.append(measure[i])
        if out:
            measures.append(out)
        measure = defaultdict(list)
        not_rests = [False for _ in range(n_spines)]

    lines = humdrum_contents.splitlines()
    n_spines = lines[0].count("\t") + 1
    measures = []
    measure = defaultdict(list)
    # Preserve the barline token (e.g. "=" or "=438") from each measure boundary
    barline_tokens: t.List[str] = []
    current_barline = "="

    # not_rests indicates whether each spine contains notes (i.e., not only
    #   rests) within each measure. It is re-initialized by _init_measure().
    not_rests = [False for _ in range(n_spines)]

    for line_i, line in enumerate(lines):
        if line.startswith("="):
            # Grab the barline token from the first spine (e.g. "=438")
            current_barline = line.split("\t")[0]
            barline_tokens.append(current_barline)
            _init_measure()
            continue
        for i, token in enumerate(line.split("\t")):
            if token.startswith("="):
                raise ValueError(
                    f"Line {line_i}: barline token '=' in column {i} but not "
                    f"all columns are barlines. This usually means the input "
                    f"spines have misaligned barlines (e.g. from a mid-measure "
                    f"crop). Full line: {line!r}"
                )
            if _is_note_token(token):
                not_rests[i] = True
            measure[i].append(token)

    # (Malcolm 2023-10-20) calling _init_measure appends the final measure to the
    # output
    _init_measure()

    out = []
    n_spines = len(measures[0])
    for measure_i, measure in enumerate(measures):
        _sort_measure_spines(measure)
        delta = len(measure) - n_spines
        if delta > 0:
            for d in range(delta):
                out.append("\t".join(["*"] * (n_spines - 1 + d) + ["*^"]))
        elif delta < 0:
            # 3 -> 2
            # delta = -1
            # n_spines = 3
            #
            for d in range(-delta):
                out.append("\t".join(["*"] * (n_spines - 2 - d) + ["*v", "*v"]))
        n_spines = len(measure)
        barline = barline_tokens[measure_i - 1] if measure_i > 0 and measure_i - 1 < len(barline_tokens) else "="
        out.append("\t".join([barline] * n_spines))
        for row in list(map(list, zip(*measure))):
            out.append("\t".join(row))
    # The first row is a spurious barline that we don't in fact want
    return out[1:]
