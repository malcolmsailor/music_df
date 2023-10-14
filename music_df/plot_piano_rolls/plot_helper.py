from typing import Any, Sequence

import pandas as pd
from matplotlib import pyplot as plt

from music_df.plot_piano_rolls.plot import get_colormapping, plot_piano_roll


def plot_predictions(
    music_df: pd.DataFrame,
    feature_name: str,
    predicted_feature: Sequence[Any],
    prediction_indices: list[int],
    plot_predicted_notes_only: bool = True,
    transpose: int | None = None,
    colormapping=None,
    label_notes: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """
    Args:
        music_df: dataframe
        predicted_feature: sequence of predictions
        prediction_indices: a sequence of features mapping predictions to notes in the
            music_df
    """

    if plot_predicted_notes_only:
        # TODO: (Malcolm 2023-09-25) maybe we want to keep non-note events
        music_df = music_df.filter(items=prediction_indices, axis=0)

    if transpose is not None:
        music_df.pitch += transpose

    music_df[f"pred_{feature_name}"] = None
    for pred, i in zip(predicted_feature, prediction_indices):
        music_df.loc[i, f"pred_{feature_name}"] = pred

    if colormapping is None:
        colormapping = get_colormapping(music_df[feature_name])

    music_df["correct"] = music_df[f"pred_{feature_name}"] == music_df[feature_name]
    music_df["colors"] = [colormapping[x] for x in music_df[feature_name]]
    if label_notes:
        music_df["label_colors"] = music_df["correct"].replace(
            {True: "black", False: "red"}
        )
    music_df = music_df[music_df.type == "note"]

    plot_piano_roll(
        music_df,
        colors=music_df["colors"].to_list(),
        labels=music_df[f"pred_{feature_name}"].to_list(),
        label_colors=music_df["label_colors"].to_list(),
        ax=ax,
        title=title,
    )
