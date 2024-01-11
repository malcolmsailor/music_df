import ast
import logging
import os
import subprocess
import tempfile
from fractions import Fraction

import pandas as pd
from metricker.meter import MeterError

from music_df.humdrum_export.dur_to_kern import KernDurError
from music_df.humdrum_export.humdrum_export import df2hum

LOGGER = logging.getLogger(__name__)

HUM2PDF = os.path.join(
    os.path.dirname((os.path.realpath(__file__))),
    "..",
    "..",
    "scripts",
    "humdrum_export",
    "hum2pdf.sh",
)
HUM2PDF_NO_COLOR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))),
    "..",
    "..",
    "scripts",
    "humdrum_export",
    "hum2pdf_no_color.sh",
)


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
) -> pd.DataFrame:
    df = pd.read_csv(
        path, converters={"onset": onset_type, "release": release_type}, index_col=0
    )
    df.loc[df.type != "note", "release"] = float("nan")
    if "other" in df.columns:
        df.loc[df.type == "time_signature", "other"] = df.loc[
            df.type == "time_signature", "other"
        ].map(ast.literal_eval)
    if "color" in df.columns:
        df.loc[df.color.isna(), "color"] = ""
    return df


def run_hum2pdf(
    humdrum_path,
    pdf_path,
    make_dirs=True,
    has_colors: bool = False,
    keep_intermediate_files: bool = False,
):
    return_code = 1
    assert os.path.exists(HUM2PDF)
    assert os.path.exists(HUM2PDF_NO_COLOR)

    if make_dirs:
        dirs = os.path.dirname(pdf_path)
        if dirs:
            os.makedirs(dirs, exist_ok=True)

    try:
        if has_colors:
            cmd = ["bash", HUM2PDF, humdrum_path, pdf_path]
            if keep_intermediate_files:
                cmd.append("y")
            print("+ " + " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            if keep_intermediate_files:
                raise NotImplementedError(
                    "I need to add a flag to the underlying shell script"
                )
            subprocess.run(
                ["bash", HUM2PDF_NO_COLOR, humdrum_path, pdf_path], check=True
            )
        return_code = 0
    except subprocess.CalledProcessError:
        # (Malcolm 2023-12-25) Some part of the pipeline seems to sometimes fail when
        #   using autobeam, so we remove any line that says `!!!filter: autobeam`
        #   and then try again
        if keep_intermediate_files:
            tmp_krn_path = os.path.join(
                os.path.expanduser("~"), "tmp", "hum2pdf", "temp.krn"
            )
        else:
            _, tmp_krn_path = tempfile.mkstemp(suffix=".krn")
        try:
            with open(humdrum_path) as inf:
                contents = inf.readlines()
            with open(tmp_krn_path, "w") as outf:
                for line in contents:
                    if line.startswith("!!!filter: autobeam"):
                        continue
                    outf.write(line)

            # For unknown reasons capture_output=True in the next column seems to cause
            # verovio to experience segmentation faults
            if has_colors:
                cmd = ["bash", HUM2PDF, tmp_krn_path, pdf_path]
                if keep_intermediate_files:
                    cmd.append("y")
                print("+ " + " ".join(cmd))
                subprocess.run(cmd, check=True)
            else:
                if keep_intermediate_files:
                    raise NotImplementedError(
                        "I need to add a flag to the underlying shell script"
                    )
                subprocess.run(
                    ["bash", HUM2PDF_NO_COLOR, tmp_krn_path, pdf_path], check=True
                )
            return_code = 0
        except subprocess.CalledProcessError:
            print("hum2pdf failed, skipping")
        # else:
        #     print("autobeam failed")
        finally:
            if not keep_intermediate_files:
                os.remove(tmp_krn_path)

    # TODO: (Malcolm 2024-01-11) the test of this function is failing,
    #   work out why
    return return_code


def df_to_pdf(
    music_df: pd.DataFrame,
    pdf_path: str,
    keep_intermediate_files: bool = False,
    **df2hum_args,
):
    try:
        humdrum = df2hum(music_df, **df2hum_args)
    except (KernDurError, MeterError) as exc:
        LOGGER.warning(f"Can't plot {pdf_path} due to {exc}")
        return 37

    has_colors = "color" in music_df.columns

    with tempfile.NamedTemporaryFile(suffix=".krn") as tempf:
        with open("/Users/malcolm/tmp/testme.krn", "w") as outf:
            outf.write(humdrum)
        with open(tempf.name, "w") as outf:
            outf.write(humdrum)
            return run_hum2pdf(
                tempf.name,
                pdf_path,
                has_colors=has_colors,
                keep_intermediate_files=keep_intermediate_files,
            )
