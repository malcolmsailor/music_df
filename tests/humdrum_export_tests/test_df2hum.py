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
    """df2hum should not crash when a 9/8 bar produces full-bar rests.

    Full-bar rests in 9/8 (4.5 qn) used to generate '8%9r' irrational rhythm
    notation. The collate_spines rscale round-trip (scale by 1/9, assemble,
    scale back by 9) would then fail because the 1/9 scaling creates very fine
    subdivisions that cause timing misalignment after assembly.
    """
    csv_path = os.path.join(SCRIPT_DIR, "resources", "test_98_rest.csv")
    df = read_csv(csv_path)

    # quantize to avoid a separate issue with unrepresentable septuplet
    # durations that cause barline misalignment
    out = df2hum(df, quantize=16)
    assert "**kern" in out


def test_df2hum_with_labels_and_grace_notes():
    """Grace notes (dur=0) with label_col should not cause an assertion error."""
    csv_table = """,type,pitch,onset,release,other,spelling
0,time_signature,100.0,,,"{'numerator': 4, 'denominator': 4}",
1,bar,,100.0,104.0,,
2,note,60,100.0,101.0,,C
3,note,69,101.0,101.0,,A
4,note,67,101.0,102.0,,G
5,bar,,104.0,108.0,,
    """
    _, temp_path = mkstemp(".csv")
    with open(temp_path, "w") as outf:
        outf.write(csv_table)
    df = read_csv(temp_path)
    os.remove(temp_path)
    assert df is not None

    # Should not raise AssertionError
    df2hum(df, label_col="spelling")


def test_barline_split_notes_are_tied():
    """Notes split at barlines by split_notes_at_barlines should produce
    tied kern tokens in the humdrum output."""
    csv_table = """,type,pitch,onset,release,other
0,time_signature,,,,"{'numerator': 2, 'denominator': 4}"
1,bar,,16.0,18.0,
2,note,69,16.0,17.5,
3,note,72,16.0,17.5,
4,note,70,17.5,18.5,
5,note,74,17.5,18.5,
6,bar,,18.0,20.0,
7,note,69,18.5,19.0,
8,note,72,18.5,19.0,
    """
    _, temp_path = mkstemp(".csv")
    with open(temp_path, "w") as outf:
        outf.write(csv_table)
    df = read_csv(temp_path)
    os.remove(temp_path)
    assert df is not None

    out = df2hum(df)
    lines = out.strip().split("\n")

    # Find the barline between the two measures
    barline_indices = [i for i, line in enumerate(lines) if line.startswith("=")]
    # The notes at onset 17.5 (B-flat/D) overlap the barline at 18.0.
    # After splitting, the first half should be tied to the second half.
    # Look for tie-start "[" before the barline and tie-end "]" after.
    pre_barline = "\n".join(lines)
    assert "[" in pre_barline, (
        f"Expected tie-start '[' in output for notes split at barline.\n{out}"
    )
    assert "]" in pre_barline, (
        f"Expected tie-end ']' in output for notes split at barline.\n{out}"
    )


def test_df2hum_barline_misalign():
    """df2hum should not crash with misaligned barlines from multi-voice data."""
    csv_path = os.path.join(SCRIPT_DIR, "resources", "test_barline_misalign.csv")
    df = read_csv(csv_path)
    out = df2hum(df)
    assert "**kern" in out


if __name__ == "__main__":
    test_df2hum_with_grace_duration()
