import logging
import typing as t
from functools import cached_property
from math import isnan

import numpy as np
import pandas as pd

from music_df.humdrum_export.constants import HEX_CODES, USER_SIGNIFIERS

LOGGER = logging.getLogger(__name__)


def get_sorted_unique_vals(unique_vals: np.ndarray):
    # Wrapper to avoid NaN errors
    non_nan_vals = []
    for val in unique_vals:
        if isinstance(val, float) and isnan(val):
            yield val
        else:
            non_nan_vals.append(val)
    yield from sorted(non_nan_vals)


class ColorMapping:
    def __init__(
        self,
        df: pd.DataFrame,
        color_col: str,
        color_mask_col: t.Optional[str] = None,
        color_mapping: t.Optional[t.Mapping[t.Any, str]] = None,
        n_alpha_levels: int | None = None,
        min_alpha_level: float = 0.25,
        uncolored_val: t.Any | None = None,
    ):
        color_chars = sorted(USER_SIGNIFIERS)
        self.n_alpha_levels = n_alpha_levels
        self._internal_color_mapping = {}
        i = 0
        self._val_to_color_char = {}
        self._val_and_alpha_to_color_char = {}
        # hex_colors = {}

        if color_mask_col is not None:
            unique_vals = df.loc[df[color_mask_col], color_col].unique()
            unique_non_color_vals = df.loc[~df[color_mask_col], color_col].unique()
            for val in unique_non_color_vals:
                self._val_to_color_char[val] = ""
        else:
            unique_vals = df[color_col].unique()

        min_alpha_level_int = None
        if n_alpha_levels is not None:
            min_alpha_level_int = round(min_alpha_level * 255)
            assert n_alpha_levels < 255 - min_alpha_level

        def _add_color_with_alphas(val, hex_color, adjusted_i):
            assert n_alpha_levels is not None
            assert min_alpha_level_int is not None
            for alpha_i in range(n_alpha_levels):
                sub_char = color_chars[(adjusted_i + alpha_i) % len(color_chars)]
                if alpha_i >= len(color_chars):
                    LOGGER.warning(
                        f"Too many unique values to color, "
                        "{val} will have a repeated color"
                    )

                alpha_level = round(
                    (255 - min_alpha_level_int) * alpha_i / (n_alpha_levels - 1)
                    + min_alpha_level_int
                )
                alpha_hex = hex(alpha_level)[2:]
                sub_color = hex_color + alpha_hex
                self._val_and_alpha_to_color_char[(val, alpha_i)] = sub_char
                self._internal_color_mapping[sub_char] = sub_color

        adjusted_i = 0
        for val in get_sorted_unique_vals(unique_vals):
            if isinstance(val, float) and isnan(val):
                self._val_to_color_char[val] = ""
            else:
                adjusted_i = i * (1 if n_alpha_levels is None else n_alpha_levels)

                if adjusted_i >= len(color_chars):
                    LOGGER.warning(
                        f"Too many unique values to color, "
                        "{val} will have a repeated color"
                    )

                if color_mapping is not None:
                    hex_color = color_mapping.get(val, HEX_CODES[i])
                else:
                    hex_color = HEX_CODES[i]

                if n_alpha_levels is None:
                    char = color_chars[adjusted_i % len(color_chars)]
                    self._internal_color_mapping[char] = hex_color
                    self._val_to_color_char[val] = char
                else:
                    _add_color_with_alphas(val, hex_color, adjusted_i)

                i += 1

        if n_alpha_levels is not None and uncolored_val is not None:
            adjusted_i = i * (1 if n_alpha_levels is None else n_alpha_levels)
            _add_color_with_alphas(uncolored_val, "#000000", adjusted_i)

    @property
    def char_to_hex(self):
        return self._internal_color_mapping

    @property
    def value_to_char(self):
        if self.n_alpha_levels is None:
            return self._val_to_color_char
        else:
            return self._val_and_alpha_to_color_char

    @cached_property
    def value_to_hex(self):
        return {
            val: self.char_to_hex[char]
            for val, char in self.value_to_char.items()
            if char
        }


def color_df(
    df: pd.DataFrame,
    color_col: str,
    color_transparency_col: str | None,
    color_mapping: ColorMapping,
) -> pd.DataFrame:
    df = df.copy()
    if color_transparency_col is not None:
        assert color_mapping.n_alpha_levels is not None
        transparency_levels = df[color_transparency_col]
        transparency_levels = transparency_levels - transparency_levels.min()
        transparency_levels = (
            transparency_levels
            / transparency_levels.max()
            * (color_mapping.n_alpha_levels - 1)
        )
        transparency_levels = transparency_levels.fillna(0.0)
        transparency_levels = transparency_levels.round().astype(int)
    else:
        transparency_levels = None

    # map df[color_col] to df["color_char"]  using `value_to_char`
    if color_transparency_col is None:
        df["color_char"] = df[color_col].map(color_mapping.value_to_char)
    else:
        color_chars = []
        assert transparency_levels is not None
        for val, alpha in zip(df[color_col], transparency_levels):
            if isinstance(val, float) and isnan(val):
                color_chars.append("")
            elif isinstance(val, str) and (
                val.startswith("na ") or val.startswith("nan ")
            ):
                color_chars.append("")
            elif val is None:
                color_chars.append("")
            else:
                color_chars.append(color_mapping.value_to_char[val, alpha])
        df["color_char"] = color_chars

    return df
