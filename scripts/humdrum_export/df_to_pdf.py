import argparse

from music_df.crop_df import crop_df
from music_df.humdrum_export.pdf import df_to_pdf, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_pdf")
    parser.add_argument("--color-col", default=None, type=str)
    parser.add_argument("--start_i")
    parser.add_argument("--end_i")
    args = parser.parse_args()

    music_df = read_csv(args.input_csv)
    if args.start_i is not None or args.end_i is not None:
        music_df = crop_df(music_df, start_i=args.start_i, end_i=args.end_i)

    df_to_pdf(music_df, args.output_pdf, color_col=args.color_col)


if __name__ == "__main__":
    main()
