import sys
from dataclasses import dataclass

import pandas as pd

try:
    from omegaconf import OmegaConf
except ImportError as e:
    raise ImportError(
        "omegaconf is required for this script. "
        "Install with: pip install music_df[scripts]"
    ) from e

from music_df.chord_df import merge_annotations
from music_df.humdrum_export.pdf import df_to_pdf
from music_df.read_csv import read_csv
from music_df.script_helpers import spinning_wheel
from music_df.sort_df import sort_df


@dataclass
class Config:
    input_file1: str
    input_file2: str
    output_file: str


def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    assert len(df1) == len(df2), "DataFrames must have the same number of rows"
    assert df1[["type", "onset", "release", "pitch"]].equals(
        df2[["type", "onset", "release", "pitch"]]
    ), "DataFrames must have the same onset, release, and pitch values"


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    df1 = read_csv(config.input_file1)
    df2 = read_csv(config.input_file2)

    assert df1 is not None, f"Failed to read {config.input_file1}"
    assert df2 is not None, f"Failed to read {config.input_file2}"
    compare_dfs(df1, df2)
    df1 = sort_df(df1)
    df2 = sort_df(df2)
    df1["harmonic_analysis"] = merge_annotations(df1)
    df2["harmonic_analysis"] = merge_annotations(df2)
    labels = df1["harmonic_analysis"].copy()
    mask = df1["harmonic_analysis"] != df2["harmonic_analysis"]
    if not mask.any():
        print("No differences found")
        return
    labels.loc[mask] = labels[mask] + " " + df2["harmonic_analysis"][mask]
    color_col = mask.astype(int).replace({0: "#000000", 1: "#FF0000"})
    df1["label_col"] = labels
    df1["color_col"] = color_col

    with spinning_wheel():
        df_to_pdf(
            df1,
            config.output_file,
            label_col="label_col",
            label_color_col="color_col",
            capture_output=True,
        )
    print(f"Wrote {config.output_file}")


if __name__ == "__main__":
    main()
