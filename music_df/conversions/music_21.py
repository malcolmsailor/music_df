import pandas as pd
from music21.meter import TimeSignature
from music21.note import Note
from music21.stream import Measure, Part, Score


def music21_score_to_df(score: Score) -> pd.DataFrame:
    raise NotImplementedError
    # I started implementing this when I was thinking that music21's parsing
    #   of the Monteverdi files was somehow aligned. But then I realized that
    #   I was mistaken. See notes in Things.
    measure_onsets: list | None = None
    measure_releases: list | None = None
    for part in score[Part]:
        assert part is not None

        this_part_measures = part[Measure]
        these_onsets = []
        these_releases = []
        for measure in this_part_measures:
            breakpoint()

    time_sigs = score[TimeSignature].flatten()
    notes = score[Note].flatten()
    measures = score[Measure].flatten()
    breakpoint()
    pass
