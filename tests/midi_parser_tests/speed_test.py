import os
import sys
import tempfile

from music_df.midi_parser import midi_to_csv

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))


def main():
    paldir = "/Users/Malcolm/Dropbox/data_science/datasets/music_corpora/Palestrina/"
    for f in os.listdir(paldir):
        if not f.endswith(".mid"):
            continue
        print(f)
        _, csv_path = tempfile.mkstemp(suffix=".csv")
        midi_to_csv(os.path.join(paldir, f), csv_path)
        os.remove(csv_path)


if __name__ == "__main__":
    main()
