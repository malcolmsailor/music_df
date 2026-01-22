from typing import Literal

import pandas as pd

Mode = Literal["M", "m"]
MinorScaleType = Literal["harmonic", "natural", "melodic"]


def float_times_df(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_float_dtype(df["onset"]) and pd.api.types.is_float_dtype(
        df["release"]
    ):
        return df
    df["onset"] = df["onset"].astype(float)
    df["release"] = df["release"].astype(float)
    return df
