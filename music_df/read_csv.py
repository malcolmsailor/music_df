from fractions import Fraction
import ast

import pandas as pd


def read_csv(path: str, onset_type=float, release_type=float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["onset"] = [onset_type(o) for o in df.onset]
    df.loc[df.type == "note", "release"] = [
        release_type(o) for o in df.loc[df.type == "note", "release"]
    ]
    df.loc[df.type != "note", "release"] = float("nan")
    df.loc[df.type == "time_signature", "other"] = df.loc[
        df.type == "time_signature", "other"
    ].map(ast.literal_eval)
    return df
