import ast
from fractions import Fraction

import pandas as pd

from music_df.quantize_df import quantize_df


def fraction_to_float(x):
    if not x:
        return float("nan")
    if "/" in x:
        # Convert fraction to float
        return float(Fraction(x))

    # Handle the case for integers or other numerical strings
    return float(x)


def read_csv(
    path: str,
    onset_type=fraction_to_float,
    release_type=fraction_to_float,
    quantize_tpq: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        converters={"onset": onset_type, "release": release_type},
    )
    # df["onset"] = [onset_type(o) for o in df.onset]
    # df.loc[df.type == "note", "release"] = [
    #     release_type(o) for o in df.loc[df.type == "note", "release"]
    # ]
    df.loc[df.type != "note", "release"] = float("nan")
    df.loc[df.type == "time_signature", "other"] = df.loc[
        df.type == "time_signature", "other"
    ].map(ast.literal_eval)
    if "color" in df.columns:
        df.loc[df.color.isna(), "color"] = ""
    if quantize_tpq is not None:
        df = quantize_df(df, quantize_tpq)
    return df
