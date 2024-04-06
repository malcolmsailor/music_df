import pandas as pd
from music21.meter import TimeSignature
from music21.note import Note
from music21.stream import Measure, Part, Score

from music_df import sort_df
from music_df.add_feature import add_bar_durs


def music21_score_to_df(score: Score) -> pd.DataFrame:
    # TODO: (Malcolm 2024-03-28) remove
    # # I started implementing this when I was thinking that music21's parsing
    # #   of the Monteverdi files was somehow aligned. But then I realized that
    # #   I was mistaken. See notes in Things.
    # measure_onsets: list | None = None
    # measure_releases: list | None = None
    # # for part in score[Part]:
    # #     assert part is not None

    # #     this_part_measures = part[Measure]
    # #     these_onsets = []
    # #     these_releases = []
    # #     for measure in this_part_measures:
    # #         for note in measure[Note]:
    # #             breakpoint()

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
