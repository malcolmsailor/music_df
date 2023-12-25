import argparse

from music_df.humdrum_export.pdf import df_to_pdf, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_pdf")
    args = parser.parse_args()

    df_to_pdf(read_csv(args.input_csv), args.output_pdf)


if __name__ == "__main__":
    main()
