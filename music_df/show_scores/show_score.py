import logging
from typing import Any, Sequence

import pandas as pd

from music_df.crop_df import crop_df

LOGGER = logging.getLogger(__name__)

try:
    from df2hum.pdf import df_to_pdf
except ImportError:

    def show_score(*args, **kwargs):
        raise ImportError("Requires df2hum")

    def show_score_and_predictions(*args, **kwargs):
        raise ImportError("Requires df2hum")

else:

    def show_score(  # type:ignore
        music_df: pd.DataFrame,
        feature_name: str,
        pdf_path: str,
        csv_path: str | None = None,
    ):
        # Get most common value of the music_df[feature_name] column:
        most_common_value = music_df[feature_name].mode()[0]

        music_df = music_df.copy()

        music_df["color_mask"] = (
            (music_df[feature_name] != most_common_value)
            & (music_df[feature_name] != "na")
            & (~music_df[feature_name].isna())
        )

        if csv_path is not None:
            music_df.to_csv(csv_path, index=False)
            LOGGER.info(f"Wrote {csv_path}")
        return df_to_pdf(
            music_df,
            pdf_path,
            color_col=feature_name,
            color_mask_col="color_mask",
            uncolored_val=most_common_value,
        )

    def show_score_and_predictions(  # type:ignore
        music_df: pd.DataFrame,
        feature_name: str,
        predicted_feature: Sequence[Any],
        prediction_indices: Sequence[int],
        pdf_path: str,
        csv_path: str | None = None,
        col_type=str,
        entropy: Sequence[float] | None = None,
        n_entropy_levels: int = 4,
    ):
        music_df = music_df.copy()
        if prediction_indices is None:
            prediction_indices = range(len(predicted_feature))

        music_df[f"pred_{feature_name}"] = None
        for pred, i in zip(predicted_feature, prediction_indices):
            music_df.loc[i, f"pred_{feature_name}"] = pred

        if entropy is not None:
            music_df["entropy"] = float("nan")
            for e, i in zip(entropy, prediction_indices):
                # Flip the sign so that low entropy notes have solid color and high
                #   entropy notes are progressively more transparent
                music_df.loc[i, "entropy"] = -e
            transparency_args = {
                "n_transparency_levels": n_entropy_levels,
                "color_transparency_col": "entropy",
            }
        else:
            transparency_args = {}

        music_df = crop_df(
            music_df, start_i=min(prediction_indices), end_i=max(prediction_indices)
        )

        if feature_name not in music_df.columns:
            # Unlabeled data
            if entropy is not None:
                raise NotImplementedError
            show_score(music_df, feature_name=f"pred_{feature_name}", pdf_path=pdf_path)
            return

        music_df["correct"] = music_df[f"pred_{feature_name}"].astype(
            col_type
        ) == music_df[feature_name].astype(col_type)
        # Concatenate strings in "correct" and feature_name columns:
        music_df["correct_by_feature"] = music_df[feature_name].astype(str) + music_df[
            "correct"
        ].map({True: " (correct)", False: " (incorrect)"})

        # Whatever the most common value is, we don't color it when it is correct

        # Get most common value of the music_df[feature_name] column:
        most_common_value = music_df[feature_name].mode()[0]
        uncolored_val = f"{most_common_value} (correct)"

        music_df["color_mask"] = (
            (music_df["correct_by_feature"] != uncolored_val)
            & (music_df[feature_name] != "na")
            & (~music_df[feature_name].isna())
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
        return df_to_pdf(
            music_df,
            pdf_path,
            label_col=f"pred_{feature_name}",
            label_mask_col="label_mask",
            label_color_col="label_color",
            color_col="correct_by_feature",
            color_mask_col="color_mask",
            uncolored_val=uncolored_val,
            **transparency_args,
        )
