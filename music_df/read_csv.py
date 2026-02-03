import ast
import logging
import os
from collections import defaultdict
from copy import copy
from fractions import Fraction
from typing import Mapping

import pandas as pd
from pandas._typing import FilePath, ReadCsvBuffer

from music_df.quantize_df import quantize_df

LOGGER = logging.getLogger(__name__)


COLUMN_DTYPES = dict(
    onset=float,
    release=float,
    pitch=float,
    tie_to_next=bool,
    tie_to_prev=bool,
    grace=bool,
    voice=float,
    part=str,
    instrument=str,
    midi_instrument=float,
    type=str,
    other=object,
)


def fraction_to_float(x):
    if not x:
        return float("nan")
    if "/" in x:
        # Convert fraction to float
        return float(Fraction(x))

    # Handle the case for integers or other numerical strings
    return float(x)


def bool_col_with_nans(x):
    return {"True": True, "False": False}.get(x, False)


COLUMN_CONVERTERS = {
    "onset": fraction_to_float,
    "release": fraction_to_float,
    "tie_to_next": bool_col_with_nans,
    "tie_to_prev": bool_col_with_nans,
    "grace": bool_col_with_nans,
}


def read_csv(
    path: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    # onset_type=fraction_to_float,
    # release_type=fraction_to_float,
    quantize_tpq: int | None = None,
    column_dtypes: dict | None = None,
    column_converters: Mapping | None = None,
) -> pd.DataFrame | None:
    if column_dtypes is None:
        column_dtypes = COLUMN_DTYPES
    if column_converters is None:
        column_converters = COLUMN_CONVERTERS

    column_dtypes = copy(column_dtypes)
    for key in column_converters:
        column_dtypes.pop(key, None)

    if isinstance(path, str) and not os.path.exists(path):
        LOGGER.warning(f"{path} does not appear to exist")
        return None
    df = pd.read_csv(
        path,
        converters=column_converters,
        index_col=0,
        dtype=column_dtypes,
    )

    # df["onset"] = [onset_type(o) for o in df.onset]
    # df.loc[df.type == "note", "release"] = [
    #     release_type(o) for o in df.loc[df.type == "note", "release"]
    # ]
    df.loc[df.type != "note", "release"] = float("nan")
    if "other" in df.columns:
        df.loc[df.type == "time_signature", "other"] = df.loc[
            df.type == "time_signature", "other"
        ].map(ast.literal_eval)
    if "color" in df.columns:
        df.loc[df.color.isna(), "color"] = ""
    if quantize_tpq is not None:
        df = quantize_df(df, quantize_tpq)
    return df
