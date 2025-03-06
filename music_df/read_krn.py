"""
Provides functions for reading a humdrum .krn file into a music_df.

Requires the `totable` command-line tool, which is a small executable I wrote based
using `humlib` and which isn't yet distributed with this package. If you'd like to
use it, please contact me.
"""

import io
import logging
import os
import re
import subprocess
import tempfile

import pandas as pd

from music_df.sort_df import sort_df
from music_df.xml_parser import xml_parse
from music_df.xml_parser.parser import RepeatOptions

TOTABLE = os.getenv("TOTABLE")
LOGGER = logging.getLogger(__name__)


def _insert_initial_barline(df: pd.DataFrame) -> pd.DataFrame:
    barline = pd.DataFrame({"onset": [0], "type": ["bar"]})
    return pd.concat([barline, df]).reset_index(drop=True)


# TODO: (Malcolm 2023-12-26 update tempi
NAMED_TEMPI = {
    # More specific tempi (like "vivace assai") should come before less specific tempi
    #   (like "vivace")
    "presto": 172,
    "molto allegro": 152,
    "allegro di molto": 152,
    "allegro non troppo": 112,
    "allegro moderato": 112,
    "allegro assai": 144,
    "allegro": 120,
    "vivace assai": 144,
    "vivace": 136,
    "andante con moto": 100,
    "andante": 84,
    "largo assai": 60,
    "largo": 66,
    "lento": 72,
    "poco adagio": 72,
    "adagio": 60,
    "allegretto ma non troppo": 100,
    "allegretto": 108,
    "moderato": 100,
    "menuetto": 100,
    "slow march": 80,
    "march": 120,
    "andantino": 88,
}

# Missing tempi:
# affettuoso
# finale
# fuga
# etwas bewegt

# Aliases
NAMED_TEMPI["adatio"] = NAMED_TEMPI["adagio"]  # typo in source files
NAMED_TEMPI["minuetto"] = NAMED_TEMPI["menuetto"]
NAMED_TEMPI["mneuetto"] = NAMED_TEMPI["menuetto"]
NAMED_TEMPI["vivaci assai"] = NAMED_TEMPI["vivace assai"]


def infer_bpm(krn_path: str, encoding="utf-8") -> float | None:
    tempo = None
    tempo_text = None
    with open(krn_path, encoding=encoding) as inf:
        for line in inf:
            if line.startswith("="):
                break
            if line.startswith("*"):
                m = re.match(r"\*MM(?P<bpm>\d+)", line)
                if m is not None:
                    tempo = float(m.group("bpm"))
            if line.startswith("!!!OMD:"):
                tempo_text = line[7:].strip().lower()

    if tempo is None:
        if tempo_text is not None:
            for named_tempo, bpm in NAMED_TEMPI.items():
                if named_tempo in tempo_text:
                    tempo = float(bpm)
                    break
    if tempo is None:
        if tempo_text is not None:
            LOGGER.warning(
                f"Couldn't infer bpm of {krn_path} but found tempo text {tempo_text}"
            )
        else:
            LOGGER.warning(f"Couldn't infer bpm of {krn_path}")

    return tempo


def read_krn(
    krn_path: str,
    remove_graces: bool = True,
    no_final_barline: bool = True,
    ensure_initial_barline: bool = True,
    # TODO: (Malcolm 2023-12-28) why is sort False by default?
    sort: bool = False,
    infer_tempo: bool = False,
    default_tempo: float | None = None,
    label_identifiers: str | None = None,
) -> pd.DataFrame:
    assert TOTABLE is not None, "TOTABLE environment variable undefined"
    totable_cmd = [TOTABLE, krn_path]
    if label_identifiers is not None:
        totable_cmd.append(label_identifiers)

    result = subprocess.run(
        totable_cmd, check=True, capture_output=True
    ).stdout.decode()

    df = pd.read_csv(io.StringIO(result), sep="\t")
    assert df["onset"].is_monotonic_increasing, (
        f"{krn_path}: onsets are not monotonicaly increasing"
    )

    df.attrs["score_name"] = krn_path
    if remove_graces:
        df = df[(df.type != "note") | (df.release > df.onset)].reset_index(drop=True)
    # Kern files often contain a final barline, which we don't generally need
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
    if infer_tempo:
        # TODO: (Malcolm 2023-12-27) re-encode kern files to utf-8 so we don't need to
        #   set encoding here?
        bpm = infer_bpm(krn_path, encoding="cp1252")
        if bpm is None and default_tempo is not None:
            # TODO: (Malcolm 2023-12-27) set tempo heuristically?
            bpm = default_tempo
        if bpm is not None:
            df = pd.concat(
                [pd.DataFrame([{"type": "tempo", "onset": 0.0, "tempo": bpm}]), df],
                ignore_index=True,
            )

    if sort:
        sort_df(df, inplace=True)
    return df


def read_krn_via_xml(
    krn_path: str, expand_repeats: RepeatOptions = "yes"
) -> pd.DataFrame:
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
