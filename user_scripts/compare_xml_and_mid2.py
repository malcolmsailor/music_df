import argparse
import math
import os
import pdb
import sys
import traceback
import warnings
from tempfile import mkstemp

from music_df.midi_parser import df_to_midi
from music_df.read import read


def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


def do_xml(xml_path):
    xml_df = read(xml_path)

    (_, mid_path) = mkstemp(suffix=".mid")

    try:
        df_to_midi(xml_df, mid_path)
        mid_df = read(mid_path)
    finally:
        os.remove(mid_path)

    xml_df = xml_df[xml_df.type == "note"].reset_index(drop=True)
    mid_df = mid_df[mid_df.type == "note"].reset_index(drop=True)

    assert len(xml_df) == len(mid_df)

    assert (xml_df["pitch"] == mid_df["pitch"]).all()
    assert all(math.isclose(x, y) for (x, y) in zip(xml_df["onset"], mid_df["onset"]))
    assert all(
        math.isclose(x, y) for (x, y) in zip(xml_df["release"], mid_df["release"])
    )


# TODO: (Malcolm 2023-12-31)
# I'm running python compare_xml_and_mid2.py $( fd -e xml . ~/datasets/YCAC-1.0-quantized/MIDI/Composers/Handel/ )
# but it's failing on the spaces in the filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_files", nargs="+")
    args = parser.parse_args()

    breakpoint()

    warnings.filterwarnings("ignore", message="note_off event")
    for xml_file in args.xml_files:
        print(xml_file)
        do_xml(xml_file)
