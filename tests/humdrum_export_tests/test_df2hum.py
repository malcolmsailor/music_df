import io
import os
import random
from tempfile import mkstemp

import pandas as pd

from music_df.humdrum_export import df2hum
from music_df.humdrum_export.constants import USER_SIGNIFIERS
from music_df.humdrum_export.pdf import run_hum2pdf
from music_df.quantize_df import quantize_df
from music_df.read_csv import read_csv
from music_df.split_notes import split_notes_at_barlines
from music_df.xml_parser import xml_parse

TEST_FILE = os.getenv("TEST_FILE")

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
CSV_PATH_WITH_GRACE = os.path.join(
    SCRIPT_DIR, "resources", "test_df_with_q_duration.csv"
)


def test_df2hum():
    df = xml_parse(TEST_FILE)
    colors = random.choices(list(USER_SIGNIFIERS), k=len(df))
    # df["color"] = colors
    # df.loc[df.type != "note", "color"] = ""

    # boolean mask indicating values of df["spelling"] not in {"A", "B", "C#"}
    df["label_mask"] = ~df["spelling"].isin({"A", "B", "C#"})
    df["color_mask"] = ~df["spelling"].isin({"C", "D", "E", "F", "G"})
    df["label_color"] = "#000000"
    df.loc[df["label_mask"], "label_color"] = "#FF0000"

    out = df2hum(
        df,
        label_col="spelling",
        label_mask_col="label_mask",
        label_color_col="label_color",
        color_col="spelling",
        color_mask_col="color_mask",
    )
    with open("temp.tsv", "w") as outf:
        outf.write(out)
    run_hum2pdf("temp.tsv", "temp.pdf")


def test_df2hum_with_grace_duration():
    df = read_csv(CSV_PATH_WITH_GRACE)
    assert df is not None
    out = df2hum(df)
    print(out)


def test_df2hum_with_overhang():
    csv_table = """,type,pitch,onset,release,tie_to_next,tie_to_prev
0,bar,,0.0,4.0,,
0,note,60,0.0,4.0,,
0,note,64,0.0,4.001,,
0,note,67,0.0,12.0,,
0,bar,,4.0,8.0,,
0,note,72,7.999,12.0,,
0,bar,,8.0,12.0,,
0,note,76,9.0,9.001,,
0,bar,,12.0,16.0,,
    """
    _, temp_path = mkstemp(".csv")
    with open(temp_path, "w") as outf:
        outf.write(csv_table)
    df = read_csv(temp_path)
    os.remove(temp_path)
    assert df is not None

    # This command should run without an error
    df2hum(df, quantize=16)


def test_df2hum_with_98_rest():
    csv_table = """,type,pitch,onset,release,tie_to_next,tie_to_prev
0,bar,,0.0,4.0,,
0,note,60,0.0,4.0,,
0,note,64,0.0,4.001,,
0,note,67,0.0,12.0,,
0,bar,,4.0,8.0,,
0,note,72,7.999,12.0,,
0,bar,,8.0,12.0,,
0,note,76,9.0,9.001,,
0,bar,,12.0,16.0,,
    """
    _, temp_path = mkstemp(".csv")
    with open(temp_path, "w") as outf:
        outf.write(csv_table)
    df = read_csv(temp_path)
    os.remove(temp_path)


if __name__ == "__main__":
    test_df2hum_with_grace_duration()
