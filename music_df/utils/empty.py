import pandas as pd


def no_notes_in_df(df: pd.DataFrame):
    if not len(df):
        return True
    if "type" in df.columns and not len(df[df.type == "note"]):
        return True
    return False
