import pandas as pd
from music_df.conversions.ms3 import ms3_to_df


def test_ms3_to_df():
    # (Malcolm 2023-10-11) This test currently does nothing except check that
    #   the conversion runs without an exception
    # TODO: (Malcolm 2023-10-11) update the path
    TEST_PATH = "/Users/malcolm/google_drive/python/third_party/dcml_corpora/corelli/notes/op03n10d.tsv"
    ms3_df = pd.read_csv(TEST_PATH, sep="\t")
    music_df = ms3_to_df(ms3_df)
