import os
import subprocess
import io
import tempfile

import pandas as pd

from music_df.sort_df import sort_df

TOTABLE = os.getenv("TOTABLE")


def _insert_initial_barline(df: pd.DataFrame) -> pd.DataFrame:
    barline = pd.DataFrame({"onset": [0], "type": ["bar"]})
    return pd.concat([barline, df]).reset_index(drop=True)


def read_krn(
    krn_path: str,
    remove_graces: bool = True,
    no_final_barline: bool = True,
    ensure_initial_barline: bool = True,
    sort: bool = False,
) -> pd.DataFrame:
    result = subprocess.run(
        [TOTABLE, krn_path], check=True, capture_output=True
    ).stdout.decode()
    df = pd.read_csv(io.StringIO(result), sep="\t")
    df.attrs["score_name"] = krn_path
    if remove_graces:
        df = df[(df.type != "note") | (df.release > df.onset)].reset_index(
            drop=True
        )
    # Kern files often contain a final barline, which we don't need
    if no_final_barline and df.iloc[-1]["type"] == "bar":
        df = df.iloc[:-1]
    # On the other hand, we *do* want an initial barline (helps us calculate
    # whether the score starts with a pickup)
    if ensure_initial_barline and df.iloc[0]["type"] != "bar":
        df = _insert_initial_barline(df)
    # TOTABLE doesn't give bar releases, so we calculate them here
    bar_releases = df.loc[df.type == "bar", "onset"].iloc[1:].to_list() + [
        df[df.type == "note"].iloc[-1]["release"]
    ]
    df.loc[df.type == "bar", "release"] = bar_releases
    # TODO should we sort here?
    if sort:
        sort_df(df, inplace=True)
    return df


def read_krn_via_xml(krn_path: str, expand_repeats="yes") -> pd.DataFrame:
    try:
        from xml_to_note_table.parser import parse as xml_parse  # type:ignore
    except ImportError:
        raise ValueError("Running this function requires `xml_to_note_table`")

    result = subprocess.run(
        ["hum2xml", krn_path], check=True, capture_output=True
    ).stdout.decode()
    _, temp_path = tempfile.mkstemp(suffix=".xml")
    try:
        with open(temp_path, "w") as outf:
            outf.write(result)
        return xml_parse(temp_path, expand_repeats=expand_repeats)
    finally:
        os.remove(temp_path)
