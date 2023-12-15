import io
import subprocess
from fractions import Fraction
from zipfile import ZipFile

import pandas as pd

from music_df.xml_parser.parser import LIMIT_DENOMINATOR, parse_mxl_metadata

remap_time_column = lambda x: Fraction(x).limit_denominator(LIMIT_DENOMINATOR)


def run_scorext(xml_path: str) -> pd.DataFrame:
    if xml_path.endswith("mxl"):
        archive = ZipFile(xml_path)
        musicxml_path = parse_mxl_metadata(archive)
        contents = archive.read(musicxml_path)
        xml_result = subprocess.run(
            ["musicxml2hum"], capture_output=True, check=True, input=contents
        )
    else:
        with open(xml_path) as inf:
            xml_result = subprocess.run(
                ["musicxml2hum"], capture_output=True, check=True, stdin=inf
            )
    scorext_out = subprocess.run(
        ["scorext", "-nL"],
        input=xml_result.stdout,
        capture_output=True,
        check=True,
    ).stdout.decode()
    # scorext_out = sh.scorext("-nL", krn_path)
    df = pd.read_csv(io.StringIO(scorext_out), sep="\t")
    # scorext returns -12988 as the pitch when there is a rest together with a
    #   note, e.g.
    # **kern
    # 4r 4G
    # *-
    df = df[df.MIDI >= 0]
    # score_ext rounds onsets to 3 decimal digits; this causes problems when
    #   integrated with other code since, e.g., a release at 1.667 can go after
    #   an onset at 1.66666666. The best solution seems to be to use fractions.
    df["START"] = df["START"].map(remap_time_column)
    df["END"] = df["END"].map(remap_time_column)
    df = df[df.MIDI >= 0]
    df.rename(
        {"MIDI": "pitch", "START": "onset", "END": "release"},
        axis=1,
        inplace=True,
    )
    return df


if __name__ == "__main__":
    import os

    df = run_scorext(
        os.path.join(
            os.path.dirname((os.path.realpath(__file__))),
            "..",
            "resources",
            "temp.xml",
        )
    )
    breakpoint()
