import pandas as pd


def float_times_df(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_float_dtype(df["onset"]) and pd.api.types.is_float_dtype(
        df["release"]
    ):
        return df
    df["onset"] = df["onset"].astype(float)
    df["release"] = df["release"].astype(float)
    return df
