import os
import random

from music_df.humdrum_export import df2hum
from music_df.humdrum_export.constants import USER_SIGNIFIERS
from music_df.humdrum_export.pdf import run_hum2pdf
from music_df.xml_parser import xml_parse

TEST_FILE = os.getenv("TEST_FILE")


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
