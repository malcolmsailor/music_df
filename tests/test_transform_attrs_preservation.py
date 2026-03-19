"""Test that every registered transform preserves df.attrs."""

import inspect

import numpy as np
import pandas as pd
import pytest

from music_df.transforms import TRANSFORMS, _ensure_transforms_loaded


def _make_representative_df():
    """Build a minimal music_df that all transforms can process."""
    return pd.DataFrame(
        {
            "type": [
                "time_signature", "bar", "note", "note", "note", "note",
                "bar", "note", "note", "note", "note",
            ],
            "pitch": [
                np.nan, np.nan, 60, 64, 67, 72,
                np.nan, 60, 64, 67, 72,
            ],
            "onset": [
                0.0, 0.0, 0.0, 0.0, 1.0, 2.0,
                4.0, 4.0, 4.0, 5.0, 6.0,
            ],
            "release": [
                np.nan, 4.0, 1.0, 2.0, 3.0, 4.0,
                8.0, 5.0, 6.0, 7.0, 8.0,
            ],
            "track": [
                0, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 1,
            ],
            "other": [
                {"numerator": 4, "denominator": 4},
                None, None, None, None, None,
                None, None, None, None, None,
            ],
        }
    )


# Transforms that need special kwargs or setup
SPECIAL_KWARGS = {
    "slice_df": {"slice_boundaries": [2.0]},
    "crop_df": {"start_time": 0.0, "end_time": 4.0},
    "slice_into_uniform_steps": {"step_dur": 1.0, "quantize_tpq": 4},
    "subdivide_notes": {"grid_size": 1.0},
}

# Transforms that require a specially prepared input
SKIP_TRANSFORMS = {
    # Requires original_note_id column from salami_slice
    "undo_salami_slice",
}


def _get_transform_names():
    _ensure_transforms_loaded()
    return sorted(TRANSFORMS.keys())


@pytest.mark.parametrize("name", _get_transform_names())
def test_transform_preserves_attrs(name):
    if name in SKIP_TRANSFORMS:
        pytest.skip(f"{name} requires special input")

    df = _make_representative_df()
    df.attrs["_test_marker"] = True

    kwargs = SPECIAL_KWARGS.get(name, {})

    # Check the transform doesn't require positional args we haven't provided
    sig = inspect.signature(TRANSFORMS[name])
    params = {
        k: v
        for k, v in sig.parameters.items()
        if k != "df" and v.default is inspect.Parameter.empty
    }
    missing = set(params) - set(kwargs)
    if missing:
        pytest.skip(f"{name} has required params {missing} not in SPECIAL_KWARGS")

    result = TRANSFORMS[name](df, **kwargs)
    assert result.attrs.get("_test_marker") is True, (
        f"{name} dropped df.attrs (missing _test_marker)"
    )
