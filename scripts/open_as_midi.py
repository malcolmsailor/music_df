import argparse
import os
import subprocess
from tempfile import mkstemp

from music_df.midi_parser.parser import df_to_midi
from music_df.read import read


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    music_df = read(args.input_file)
    assert music_df is not None
    _, tmpfile = mkstemp(suffix=".mid")
    df_to_midi(music_df, tmpfile)

    subprocess.run(["open", tmpfile])
    input("Press enter to delete temp file...")
    os.remove(tmpfile)


if __name__ == "__main__":
    main()
