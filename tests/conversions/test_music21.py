import pytest
from music21 import corpus
from music21.stream import Score

from music_df.conversions.music_21 import music21_score_to_df


@pytest.mark.skip(reason="not_implemented")
def test_music21_to_df():
    score = corpus.parse("madrigal.5.3.mxl")
    assert isinstance(score, Score)
    df = music21_score_to_df(score)
