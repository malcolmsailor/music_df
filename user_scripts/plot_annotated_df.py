import argparse

from music_df.chord_df import keep_new_elements_only, merge_annotations
from music_df.humdrum_export.pdf import df_to_pdf
from music_df.read_csv import read_csv
from music_df.script_helpers import spinning_wheel
from music_df.sort_df import sort_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    annotated_df = read_csv(args.input_path)

    annotated_df = sort_df(annotated_df)
    annotated_df["harmonic_analysis"] = merge_annotations(annotated_df)

    with spinning_wheel():
        df_to_pdf(
            annotated_df,
            args.output_path,
            label_col="harmonic_analysis",
            capture_output=True,
        )
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
