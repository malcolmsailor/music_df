import logging
from typing import Any, Sequence

import pandas as pd
from df2hum.pdf import df_to_pdf

from music_df.crop_df import crop_df

LOGGER = logging.getLogger(__name__)


def show_score():
    pass


def show_score_and_predictions(
    music_df: pd.DataFrame,
    feature_name: str,
    predicted_feature: Sequence[Any],
    prediction_indices: Sequence[int],
    pdf_path: str,
    csv_path: str | None = None,
    col_type=str,
):
    # TODO: (Malcolm 2023-09-29) allow cropping df to predicted indices but keep
    #   immediately preceding barline and time signature
    music_df = music_df.copy()
    music_df = crop_df(
        music_df, start_i=min(prediction_indices), end_i=max(prediction_indices)
    )
    if prediction_indices is None:
        prediction_indices = range(len(predicted_feature))

    music_df[f"pred_{feature_name}"] = None
    for pred, i in zip(predicted_feature, prediction_indices):
        music_df.loc[i, f"pred_{feature_name}"] = pred

    music_df["correct"] = music_df[f"pred_{feature_name}"].astype(col_type) == music_df[
        feature_name
    ].astype(col_type)
    # Concatenate strings in "correct" and feature_name columns:
    music_df["correct_by_feature"] = music_df["correct"].astype(col_type) + music_df[
        feature_name
    ].astype(col_type)

    # Whatever the most common value is, we don't color it when it is correct

    # Get most common value of the music_df[feature_name] column:
    most_common_value = music_df[feature_name].mode()[0]
    music_df["color_mask"] = (
        music_df["correct_by_feature"] != f"{True}{most_common_value}"
    )

    # Only label incorrect notes
    music_df["label_mask"] = ~music_df["correct"]

    music_df["label_color"] = music_df["correct"].map(
        {True: "#000000", False: "#FF0000"}
    )

    assert len(music_df.loc[music_df["label_mask"], "label_color"].unique()) == 1

    if csv_path is not None:
        music_df.to_csv(csv_path, index=False)
        LOGGER.info(f"Wrote {csv_path}")
    df_to_pdf(
        music_df,
        pdf_path,
        label_col=f"pred_{feature_name}",
        label_mask_col="label_mask",
        label_color_col="label_color",
        color_col="correct_by_feature",
        color_mask_col="color_mask",
    )
