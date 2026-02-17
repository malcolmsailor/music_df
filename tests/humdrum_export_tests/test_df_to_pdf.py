import os
import tempfile

import numpy as np

from music_df.add_feature import infer_barlines
from music_df.humdrum_export.pdf import df_to_pdf, read_csv

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_output")

CSV_PATH = os.path.join(SCRIPT_DIR, "resources", "test_df.csv")
CSV_PATH_WITH_GRACE = os.path.join(
    SCRIPT_DIR, "resources", "test_df_with_q_duration.csv"
)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def test_read_csv_without_index_column():
    """read_csv should work for CSVs saved with index=False (no unnamed index column)."""
    df_with_index = read_csv(CSV_PATH)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df_with_index.to_csv(f.name, index=False)
        df_without_index = read_csv(f.name)
    os.unlink(f.name)
    assert "onset" in df_without_index.columns


def test_df_to_pdf_no_color():
    df = read_csv(CSV_PATH)
    df = infer_barlines(df)
    df_to_pdf(df, os.path.join(OUTPUT_DIR, "no_color.pdf"))


def test_df_to_pdf_color():
    df = read_csv(CSV_PATH)
    df = infer_barlines(df)
    # Choose a random range of numbers to test interpolation
    df["transparency"] = np.random.random() * 5.0 - 2.3
    return_code = df_to_pdf(
        df,
        os.path.join(OUTPUT_DIR, "color.pdf"),
        keep_intermediate_files=False,
        color_col="pitch",
        color_transparency_col="transparency",
        n_transparency_levels=5,
    )
    assert not return_code, f"Got non-zero return code {return_code}"


def test_df_to_pdf_with_grace_duration():
    df = read_csv(CSV_PATH_WITH_GRACE)
    return_code = df_to_pdf(df, os.path.join(OUTPUT_DIR, "q.pdf"))
    assert not return_code
