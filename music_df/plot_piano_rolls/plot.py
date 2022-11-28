import sys

from typing import Dict, Iterable, Optional, Any, List, Union
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

# from chord_tones_data.constants import END_TOKEN, PAD_TOKEN, START_TOKEN


def add_line_breaks(text, line_width):
    """It doesn't seem to be easy to wrap text in boxes in matplotlib so wrote
    up this function quickly as a hack.
    """
    out = []
    candidate_i = 0
    last_split_i = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            if i - last_split_i > line_width:
                if candidate_i > last_split_i:
                    out.append(text[last_split_i:candidate_i])
                    last_split_i = candidate_i + 1
            candidate_i = i
        i += 1
    out.append(text[last_split_i:])
    return "\n".join(out)


def format_title(title: str):
    try:
        title, offset = title[:-1].rsplit("(", maxsplit=1)
    except ValueError:
        return title
    return "\n".join([title, offset])


def add_piano_roll_background(
    ax: matplotlib.axes.Axes,
    tet: int = 12,
    black_keys: Iterable[int] = (1, 3, 6, 8, 10),
    # consecutive_white_keys should contain the *higher* of any two consecutive
    # white keys
    consecutive_white_keys: Iterable[int] = (0, 5),
):
    colors = ("white", "gainsboro")
    low, hi = map(int, ax.get_ylim())
    begin, end = ax.get_xlim()
    z = {"rect": 1, "line": 2}
    for pitch in range(low, hi - 1):
        color = colors[pitch % tet in black_keys]
        # For some reason that I don't understand using the "Rectangle" patch
        # doesn't seem to work
        rect = matplotlib.patches.Polygon(
            xy=[
                [begin, pitch],
                [end, pitch],
                [end, pitch + 1],
                [begin, pitch + 1],
            ],
            color=color,
            zorder=z["rect"],
        )
        ax.add_patch(rect)
        if pitch % tet in consecutive_white_keys and pitch != low:
            line = matplotlib.lines.Line2D(
                [begin, end],
                [pitch, pitch],
                color="gainsboro",
                linewidth=1,
                zorder=z["line"],
            )
            ax.add_line(line)


def add_note(
    ax, note: pd.Series, color="blue", label=None, label_color=None, number=None
):
    pitch = note.pitch
    begin = note.onset
    end = note.release
    z = 3
    rect = matplotlib.patches.Polygon(
        xy=[
            [begin, pitch],
            [end, pitch],
            [end, pitch + 1],
            [begin, pitch + 1],
        ],
        color=color,
        zorder=z,
    )
    ax.add_patch(rect)
    if label is not None:
        ax.text(begin, pitch + 1.2, label, color=label_color, zorder=4)
    if number is not None:
        ax.text(
            begin,
            pitch - 1.8,
            f"{number}:{pitch}",
            color="gray",
            zorder=3.5,
        )


def plot_piano_roll(
    df: pd.DataFrame,
    colors: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[Any]] = None,
    label_colors: Optional[Iterable[str]] = None,
    number_notes: bool = False,
    ax=None,
    subplots_args=None,
    show_axes=True,
    xticks=True,
    show=False,
    title=None,
    legend: Optional[Dict[str, Any]] = None,
):
    if "type" in df.columns and (df.type != "note").any():
        raise ValueError("df should only have 'note' events")
    if labels is not None:
        labels = [str(label) for label in labels]
    begin = df.onset.min()
    end = df.release.max()
    low = df.pitch.min()
    hi = df.pitch.max() + 1
    if ax is None:
        fig, ax = plt.subplots(
            **({} if subplots_args is None else subplots_args)
        )
    ax.set_ylim(float(low), float(hi))
    ax.set_xlim(float(begin), float(end))
    ax.set_ylabel("Midi pitch")
    if not xticks:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
    if not show_axes:
        plt.axis("off")
    add_piano_roll_background(ax)
    for i, (row_i, note) in enumerate(df.iterrows()):
        add_note(
            ax,
            note,
            (colors[i] if colors is not None else None),
            (labels[i] if labels is not None else None),
            (label_colors[i] if label_colors is not None else None),
            number=row_i if number_notes else None,
        )
    if title is not None:
        ax.set_title(format_title(title))
    if legend is not None:
        handles = []
        for label, color in legend.items():
            patch = matplotlib.patches.Patch(color=color, label=label)
            handles.append(patch)
        # put legend at bottom of plot
        # after https://stackoverflow.com/a/4701285/10155119
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95]
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.075),
            ncol=len(handles),
            handles=handles,
        )

    if show:
        plt.show()


def get_colormapping(feature, left_offset=0, right_offset=1):
    unique = pd.Series(feature).unique()
    unique.sort()
    viridis = matplotlib.cm.get_cmap("viridis")
    if unique.dtype == bool:
        out = {True: viridis(0.0), False: viridis(0.5)}
        return out
    out = {}
    for i, item in enumerate(unique):
        scale = 1 - left_offset - (1 - right_offset)
        assert scale > 0
        out[item] = viridis(
            (i / (max(1, len(unique) - 1))) * scale + left_offset
        )
    return out


def plot_piano_roll_and_feature(
    df: pd.DataFrame,
    feature: List[Any],
    featuremapping=None,
    label_notes=True,
    colormapping=None,
    ax=None,
    title=None,
    transparencies=None,
    legend=False,
):
    # TODO implement legend?
    if featuremapping is not None:
        feature = [featuremapping[f] for f in feature]
    if colormapping is None:
        colormapping = get_colormapping(feature)
    colors = [colormapping[item] for item in feature]
    if transparencies is not None:
        colors = [
            color[:3] + (t,) for (color, t) in zip(colors, transparencies)
        ]
    plot_piano_roll(
        df,
        colors,
        title=title,
        labels=feature if label_notes else None,
        ax=ax,
        legend=colormapping,
    )


RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def apply_text_annotations(
    text_annotations: List[str],
    fig: matplotlib.figure.Figure,
    ax: plt.Axes,
    red_text_flag: bool = False,
):
    bb = ax.get_window_extent()
    bb_coords = fig.transFigure.inverted().transform(bb)
    plt.figtext(
        x=bb_coords[0, 0],
        y=0.01,
        s="\n".join([add_line_breaks(t, 50) for t in text_annotations]),
        color="red" if red_text_flag else "black",
    )


def plot_feature_and_accuracy(
    events: List[str],
    target_feature: List[Any],
    pred_feature: List[Any],
    featuremapping=None,
    label_notes=True,
    colormapping=None,
    predmapping=None,
    title=None,
    number_notes: bool = False,
    transpose: Optional[int] = None,
    ax=None,
    fig=None,
    mpl_text=True,
    pred_start_token: Optional[Any] = None,
    start_token="<START>",
    end_token="<STOP>",
):
    """
    keyword args:
        predmapping: if predictions have different labels from targets, we can
            remap them with this dict.
    """
    # Not sure whether this package is the right place for this function.
    if mpl_text:
        red = green = reset = ""
    else:
        red = RED
        green = GREEN
        reset = RESET

    df = events

    if transpose is not None:
        df.pitch += transpose
    if predmapping is not None:
        pred_feature = [predmapping[p] for p in pred_feature]
    text_annotations = []
    if pred_start_token is not None:
        if pred_feature[0] != pred_start_token:
            text_annotations.append(
                f"{red}First symbol {pred_feature[0]} doesn't "
                f"match start symbol {pred_start_token}{reset}"
            )
        pred_feature = pred_feature[1:]
    elif target_feature[0] == start_token:
        if pred_feature[0] == start_token:
            text_annotations.append(f"{green}Start symbol matches{reset}")
        else:
            text_annotations.append(f"{red}Start symbol doesn't match{reset}")
        target_feature = target_feature[1:]
        pred_feature = pred_feature[1:]
    if target_feature[-1] == end_token:
        if pred_feature[-1] == end_token:
            text_annotations.append(
                f"{green}Prediction ends with end symbol{reset}"
            )
        else:
            text_annotations.append(
                f"{red}Prediction doesn't end with end symbol{reset}"
            )
    correct = [(t == p, t) for (t, p) in zip(target_feature, pred_feature)]
    correct.extend([(False, t) for t in target_feature[len(pred_feature) :]])
    len_delta = len(pred_feature) - len(target_feature)
    if len_delta > 0:
        text_annotations.append(
            f"{red}Predicted target has {len_delta} excess symbols{reset}"
        )
    elif len_delta < 0:
        text_annotations.append(
            f"{red}Predicted target has {-len_delta} too few symbols{reset}"
        )

    if featuremapping is not None:
        pred_feature = [featuremapping[f] for f in pred_feature]
    if colormapping is None:
        colormapping = get_colormapping(correct)
    # we truncate by len(df) in case the inputs were truncated such that
    # there are note_on events with no associated note_offs
    colors = [colormapping[item] for item in correct][: len(df)]
    if label_notes:
        labels = pred_feature.copy()
        labels.extend(
            [
                "None"
                for _ in range(max(0, len(target_feature) - len(pred_feature)))
            ]
        )
        labels = labels[: len(df)]
        label_colors = ["black" if b else "red" for (b, _) in correct]
    else:
        labels = None
    if labels[0] == "start":
        labels = labels[1:]
        label_colors = label_colors[1:]
        colors = labels[1:]
    plot_piano_roll(
        df,
        colors,
        labels=labels,
        label_colors=label_colors,
        number_notes=number_notes,
        title=title,
        ax=ax,
    )
    if mpl_text:
        apply_text_annotations(text_annotations, fig, ax)
    else:
        for t in text_annotations:
            print(t)
    return df


def plot_feature_and_accuracy_token_class(
    events: Union[pd.DataFrame, List[str]],
    target_feature: List[Any],
    pred_feature: List[Any],
    # pred_feature_probs: Optional[List[float]] = None,
    featuremapping=None,
    label_notes=True,
    colormapping=None,
    predmapping=None,
    title=None,
    number_notes: bool = False,
    transpose: Optional[int] = None,
    ax=None,
    fig=None,
    mpl_text=True,
    pad_token: str = "<PAD>",
    start_token: str = "<START>",
    end_token: str = "<STOP>",
):
    # Not sure whether this package is the right place for this function.
    """
    keyword args:
        predmapping: if predictions have different labels from targets, we can
            remap them with this dict.
    """
    if mpl_text:
        red = green = reset = ""
    else:
        red = RED
        green = GREEN
        reset = RESET
    red_text_flag = False

    if isinstance(events, list):
        raise NotImplementedError
    else:
        df = events

    if transpose is not None:
        df.pitch += transpose
    if predmapping is not None:
        pred_feature = [predmapping[p] for p in pred_feature]
    text_annotations = []
    if target_feature[0] == start_token:
        if pred_feature[0] == start_token:
            text_annotations.append(f"{green}Start symbol matches{reset}")
        else:
            text_annotations.append(f"{red}Start symbol doesn't match{reset}")
        target_feature = target_feature[1:]
        pred_feature = pred_feature[1:]
    if target_feature[-1] == end_token:
        if pred_feature[-1] == end_token:
            text_annotations.append(
                f"{green}Prediction ends with end symbol{reset}"
            )
        else:
            text_annotations.append(
                f"{red}Prediction doesn't end with end symbol{reset}"
            )
            red_text_flag = True
    if featuremapping is not None:
        pred_feature = [featuremapping[f] for f in pred_feature]
    target_nonpad = []
    pred_nonpad = []
    pred_pad = []
    target_pad_count = 0
    pred_pad_count = 0
    for token1, token2 in zip(target_feature, pred_feature):
        if token1 == pad_token:
            target_pad_count += 1
            pred_pad.append(token2)
            if token1 == token2:
                pred_pad_count += 1
        else:
            target_nonpad.append(token1)
            pred_nonpad.append(token2)
    correct = [(t == p, t) for (t, p) in zip(target_nonpad, pred_nonpad)]
    if target_pad_count == pred_pad_count:
        text_annotations.append(f"{green}All pad tokens match{reset}")
    else:
        text_annotations.append(
            f"{red}{pred_pad_count}/{target_pad_count} pad tokens match{reset}"
        )
        red_text_flag = True
    if colormapping is None:
        colormapping = get_colormapping(correct)
    # we truncate by len(df) in case the inputs were truncated such that
    # there are note_on events with no associated note_offs
    colors = [colormapping[item] for item in correct][: len(df)]
    if label_notes:
        labels = pred_nonpad.copy()
        label_colors = ["black" if b else "red" for (b, _) in correct]
    else:
        labels = None
    if labels[0] == "start":
        labels = labels[1:]
        label_colors = label_colors[1:]
        colors = labels[1:]
    plot_piano_roll(
        df,
        colors,
        labels=labels,
        label_colors=label_colors,
        number_notes=number_notes,
        title=title,
        ax=ax,
    )
    if mpl_text:
        apply_text_annotations(text_annotations, fig, ax, red_text_flag)
    else:
        for t in text_annotations:
            print(t)
    return df
