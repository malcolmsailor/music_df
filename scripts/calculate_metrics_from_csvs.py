import argparse
import ast

import numpy as np
import pandas as pd
import sklearn.metrics

DEBUG = True
if DEBUG:
    import pdb
    import sys
    import traceback

    def custom_excepthook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)

    sys.excepthook = custom_excepthook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+")
    parser.add_argument("--uniform-steps", action="store_true")
    parser.add_argument(
        "--output-file", default=None, help="We will append a row to this csv file"
    )
    parser.add_argument("--key", default=None, help="If present, first item in csv row")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.input_files) == 1:
        df = pd.read_csv(args.input_files[0])
        df = df.drop(["path", "indices"], axis=1)
        y_true = np.concatenate([np.array(ast.literal_eval(x)) for x in df["labels"]])
        y_pred = np.concatenate(
            [np.array(ast.literal_eval(x)) for x in df["predicted"]]
        )
    else:
        df = pd.read_csv(args.input_files[0]).drop(["path", "indices"], axis=1)
        for i, input_file in enumerate(args.input_files[1:], start=1):
            new_df = pd.read_csv(input_file).drop(["path", "indices"], axis=1)
            df = df.merge(
                new_df,
                left_index=True,
                right_index=True,
                suffixes=("", f"_{i}"),
                how="outer",
            )
        df = df.rename({"predicted": "predicted_0", "labels": "labels_0"}, axis=1)
        y_trues = [
            np.concatenate([np.array(ast.literal_eval(x)) for x in df[f"labels_{i}"]])
            for i in range(len(args.input_files))
        ]
        y_preds = [
            np.concatenate(
                [np.array(ast.literal_eval(x)) for x in df[f"predicted_{i}"]]
            )
            for i in range(len(args.input_files))
        ]
        y_true = ["_".join(str(x) for x in xs) for xs in zip(*y_trues)]
        y_pred = ["_".join(str(x) for x in xs) for xs in zip(*y_preds)]

    if args.uniform_steps:
        if "uniform_steps" not in df.columns:
            raise ValueError(f"No 'uniform_steps' column in {df.columns=}")

        repeats = np.concatenate(
            [np.array(ast.literal_eval(x)) for x in df["uniform_steps"]]
        )
        y_true = np.repeat(y_true, repeats)
        y_pred = np.repeat(y_pred, repeats)
    del df

    assert len(y_true) == len(y_pred)

    # precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
    #     y_true, y_pred, average="weighted"
    # )
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    print(f"{accuracy=}")
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    print(f"{balanced_accuracy=}")
    if args.output_file is not None:
        with open(args.output_file, "a") as outf:
            if args.key is None:
                outf.write(f"{accuracy},{balanced_accuracy}\n")
            else:
                outf.write(f"{args.key},{accuracy},{balanced_accuracy}\n")


if __name__ == "__main__":
    main()
