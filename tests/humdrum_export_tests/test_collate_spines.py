import os

from music_df.humdrum_export.collate_spines import collate_spines

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))

TEST_FILES = [os.path.join(SCRIPT_DIR, "resources", p) for p in ("1.krn", "2.krn")]


def test_collate_spines():
    result = collate_spines(TEST_FILES)
