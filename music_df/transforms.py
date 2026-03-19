"""
A decorator-based registry for df -> df transforms, and a function to apply
a sequence of transforms to a dataframe.

Usage:

    Registering a transform (in the module that defines it):

        from music_df.transforms import transform

        @transform
        def dedouble(df, match_releases=True, ...):
            ...

    Applying transforms:

        from music_df.transforms import apply_transforms

        steps = [
            {"quantize_df": {"tpq": 16}},
            {"merge_notes": {}},
            {"salami_slice": {}},
            {"dedouble": {"match_releases": True}},
        ]
        result = apply_transforms(df, steps)

    Listing available transforms:

        from music_df.transforms import TRANSFORMS
        print(list(TRANSFORMS))
"""

import inspect
from typing import Callable

import pandas as pd

TRANSFORMS: dict[str, Callable[..., pd.DataFrame]] = {}


def transform(
    func: Callable[..., pd.DataFrame] | None = None,
    *,
    diff_func: Callable[
        [pd.DataFrame, pd.DataFrame],
        tuple[set, set] | tuple[set, set, list[tuple[float, float]]],
    ]
    | None = None,
) -> Callable[..., pd.DataFrame] | Callable[[Callable], Callable]:
    """Register a df -> df function as a named transform.

    Supports both ``@transform`` and ``@transform(diff_func=...)``.

    When *diff_func* is provided it is attached as an attribute on the
    registered function so that callers (e.g. demo_transforms) can use a
    custom diff instead of naive tuple comparison.
    """
    def _register(f: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        if f.__name__ in TRANSFORMS:
            raise ValueError(
                f"Transform {f.__name__!r} is already registered "
                f"(from {TRANSFORMS[f.__name__].__module__})"
            )
        if diff_func is not None:
            f.diff_func = diff_func  # type: ignore[attr-defined]
        TRANSFORMS[f.__name__] = f
        return f

    if func is not None:
        return _register(func)
    return _register


def get_transform_params(name: str) -> dict[str, inspect.Parameter]:
    """Return the keyword parameters for a registered transform (excluding `df`)."""
    _ensure_transforms_loaded()
    if name not in TRANSFORMS:
        raise KeyError(f"Unknown transform {name!r}")
    sig = inspect.signature(TRANSFORMS[name])
    return {
        k: v
        for k, v in sig.parameters.items()
        if k != "df" and v.kind in (v.POSITIONAL_OR_KEYWORD, v.KEYWORD_ONLY)
    }


def apply_transforms(
    df: pd.DataFrame,
    steps: list[dict[str, dict]],
) -> pd.DataFrame:
    """Apply a sequence of transforms to a dataframe.

    Args:
        df: The input dataframe.
        steps: A list of single-key dicts mapping transform name to kwargs.
            E.g., [{"quantize_df": {"tpq": 16}}, {"dedouble": {}}]

    Returns:
        The transformed dataframe.

    >>> import pandas as pd
    >>> from music_df.transforms import transform, apply_transforms, TRANSFORMS

    We'll register a couple of toy transforms for testing:
    >>> @transform
    ... def _test_add_col(df, col_name="foo", value=1):
    ...     df = df.copy()
    ...     df[col_name] = value
    ...     return df
    ...
    >>> @transform
    ... def _test_double_col(df, col_name="foo"):
    ...     df = df.copy()
    ...     df[col_name] = df[col_name] * 2
    ...     return df
    ...

    >>> df = pd.DataFrame({"pitch": [60, 62]})
    >>> steps = [
    ...     {"_test_add_col": {"col_name": "x", "value": 10}},
    ...     {"_test_double_col": {"col_name": "x"}},
    ... ]
    >>> result = apply_transforms(df, steps)
    >>> list(result["x"])
    [20, 20]

    Cleanup:
    >>> del TRANSFORMS["_test_add_col"]
    >>> del TRANSFORMS["_test_double_col"]
    """
    _ensure_transforms_loaded()
    for step in steps:
        if len(step) != 1:
            raise ValueError(
                f"Each step must be a single-key dict, got {len(step)} keys: "
                f"{list(step.keys())}"
            )
        name, kwargs = next(iter(step.items()))
        if name not in TRANSFORMS:
            raise KeyError(
                f"Unknown transform {name!r}. Available: {sorted(TRANSFORMS.keys())}"
            )
        df = TRANSFORMS[name](df, **kwargs)
    return df


_transforms_loaded = False


_TRANSFORM_MODULES = (
    "music_df.add_feature",
    "music_df.crop_df",
    "music_df.dedouble",
    "music_df.dedouble_instruments",
    "music_df.detremolo",
    "music_df.merge_notes",
    "music_df.quantize_df",
    "music_df.remove_repeated_bars",
    "music_df.salami_slice",
    "music_df.slice_df",
    "music_df.sort_df",
    "music_df.split_notes",
)


def _ensure_transforms_loaded():
    """Import all modules that register transforms.

    Called lazily on first use of apply_transforms to avoid circular imports
    at module load time. Modules with missing optional dependencies are
    silently skipped.
    """
    global _transforms_loaded
    if _transforms_loaded:
        return
    _transforms_loaded = True
    import importlib

    for mod_name in _TRANSFORM_MODULES:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass
