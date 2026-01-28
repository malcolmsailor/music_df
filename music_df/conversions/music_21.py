"""
A function for converting music21 scores to music dataframes.
"""

import pandas as pd

try:
    from music21.meter import TimeSignature
    from music21.note import Note
    from music21.stream import Measure, Part, Score
except ImportError as e:
    raise ImportError(
        "music21 is required for this feature. "
        "Install with: pip install music_df[music21]"
    ) from e

from music_df import sort_df
from music_df.add_feature import add_bar_durs


def music21_score_to_df(score: Score) -> pd.DataFrame:
    rows = []
    time_sigs = score[TimeSignature].stream().flatten()  # type:ignore

    last_time_sig_offset = None
    for time_sig in time_sigs:
        if time_sig.offset == last_time_sig_offset:
            # Don't include duplicate time sigs across different parts
            continue

        rows.append(
            {
                "type": "time_signature",
                "onset": time_sig.offset,
                "other": {
                    "numerator": time_sig.numerator,
                    "denominator": time_sig.denominator,
                },
            }
        )
        last_time_sig_offset = time_sig.offset

    for part_i, part in enumerate(score[Part], start=1):
        notes = part[Note].stream().flatten()

        for note in notes:
            tie_to_next = False
            tie_to_prev = False

            if note.tie:
                if note.tie.type == "start":
                    tie_to_next = True
                elif note.tie.type == "stop":
                    tie_to_prev = True
                elif note.tie.type == "continue":
                    tie_to_next = True
                    tie_to_prev = True
                else:
                    raise NotImplementedError

            rows.append(
                {
                    "type": "note",
                    "onset": note.offset,
                    "release": note.offset + note.quarterLength,
                    "pitch": note.pitch.midi,
                    "tie_to_next": tie_to_next,
                    "tie_to_prev": tie_to_prev,
                    "voice": 1,  # TODO: (Malcolm 2024-03-28)
                    "part": part_i,
                    "spelling": note.pitch.name,
                }
            )

    measures = score[Measure]
    for measure in measures:
        rows.append({"type": "bar", "onset": measure.offset})

    df = pd.DataFrame(rows)
    df = sort_df(df)
    add_bar_durs(df)
    # This seems to be the easiest way of removing duplicate barlines across different parts
    df = df[~((df.type == "bar") & (df.bar_dur == 0))]
    df = df.drop("bar_dur", axis=1)
    df = df.reset_index(drop=True)
    return df
