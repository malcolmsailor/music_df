import pandas as pd
from music21 import converter
from music21.meter import TimeSignature
from music21.stream import Measure

from music_df.sort_df import sort_df


def _get_bar_row(measure: Measure):
    onset = measure.offset
    release = onset + measure.quarterLength
    return {"type": "bar", "onset": onset, "release": release}


def _get_time_sig_row(time_sig: TimeSignature):
    onset = time_sig.offset
    other = {"numerator": time_sig.numerator, "denominator": time_sig.denominator}
    return {"type": "time_signature", "onset": onset, "other": other}


def read_rntxt(input_path: str) -> pd.DataFrame:
    score = converter.parse(input_path, format="romanText")
    measures = list(score[Measure])
    bar_rows = [_get_bar_row(measure) for measure in measures]

    # We flatten so that offsets will be relative to start of score
    flat_score = score.flatten()
    time_sigs = flat_score[TimeSignature]
    time_sig_rows = [_get_time_sig_row(ts) for ts in time_sigs]

    df = pd.DataFrame(bar_rows + time_sig_rows)
    df = sort_df(df)

    return df
