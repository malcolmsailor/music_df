import glob
import os
import pdb
import sys
import traceback

from music_df.read_krn import infer_bpm
from tests.helpers_for_tests import HUMDRUM_DATA_PATH

assert HUMDRUM_DATA_PATH is not None


def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


def main():
    files = glob.glob(os.path.join(HUMDRUM_DATA_PATH, "**", "*.krn"), recursive=True)
    files = [f for f in files if not "jrp" in f]
    for f in files:
        try:
            infer_bpm(f, encoding="cp1252")
        except UnicodeDecodeError:
            print(f"WARNING: Can't decode {f}")


if __name__ == "__main__":
    main()
