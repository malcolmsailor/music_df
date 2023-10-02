from typing import Any, Sequence

import pandas as pd
from df2hum.pdf import df_to_pdf

from music_df.crop_df import crop_df


def show_score():
    pass


def show_score_and_predictions(
    music_df: pd.DataFrame,
    feature_name: str,
    predicted_feature: Sequence[Any],
    prediction_indices: Sequence[int],
    pdf_path: str,
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

    music_df["correct"] = music_df[f"pred_{feature_name}"] == music_df[feature_name]
    # Concatenate strings in "correct" and feature_name columns:
    music_df["correct_by_feature"] = (
        music_df["correct"].astype(str) + music_df[feature_name]
    )

    df_to_pdf(
        music_df, pdf_path, color_col="correct_by_feature", dont_color_values={True}
    )
