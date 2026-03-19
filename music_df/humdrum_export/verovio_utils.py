"""Utilities for safe verovio rendering."""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class VerovioLoadError(Exception):
    """Raised when verovio crashes or times out loading a humdrum string."""


def _probe_load(
    hum: str, *, options: dict | None = None, timeout: int = 30
) -> bool:
    """Try loading *hum* in a subprocess. Returns True if it succeeded."""
    with tempfile.NamedTemporaryFile(
        suffix=".krn", mode="w", delete=False
    ) as f:
        f.write(hum)
        tmp_path = f.name

    # Options affect verovio's analysis passes (e.g. beam grouping depends
    # on page width), so the probe must use the same options as the caller.
    opts_snippet = ""
    if options:
        opts_snippet = f"tk.setOptions({options!r}); "

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import verovio, sys; "
                "tk = verovio.toolkit(); "
                f"{opts_snippet}"
                f"tk.loadData(open({tmp_path!r}).read()); "
                "n = tk.getPageCount(); "
                "[tk.renderToSVG(i) for i in range(1, n + 1)]; "
                "sys.exit(0 if n > 0 else 1)",
            ],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def verovio_safe_load(tk, hum: str, *, options: dict | None = None) -> bool:
    """Load humdrum into verovio, probing in a subprocess first.

    Verovio's C++ code can segfault on certain inputs (e.g. complex beam
    patterns with many spines). A segfault in the main process kills the
    entire application, so we always probe in a subprocess first.

    If the humdrum contains ``!!!filter:`` lines and the probe crashes,
    we retry without filters (the autobeam filter is a common culprit).
    If the filterless version also crashes, we raise ``VerovioLoadError``.

    Args:
        tk: A verovio.toolkit() instance.
        hum: Humdrum string to load.
        options: Verovio options dict (as passed to ``tk.setOptions``).
            The probe subprocess must use the same options because they
            can affect analysis passes like beam grouping.

    Returns:
        True if filters were stripped (fallback was used), False otherwise.

    Raises:
        VerovioLoadError: If verovio crashes on the input even without filters.
    """
    if _probe_load(hum, options=options):
        tk.loadData(hum)
        return False

    # Probe failed — try without filters if there are any
    has_filters = "!!!filter:" in hum
    if has_filters:
        hum_no_filter = "\n".join(
            line for line in hum.split("\n") if not line.startswith("!!!filter:")
        )
        if _probe_load(hum_no_filter, options=options):
            logger.warning("Verovio crashed with filters; loading without them")
            tk.loadData(hum_no_filter)
            return True

    raise VerovioLoadError(
        "Verovio crashed loading this input"
        + (" even without filters" if has_filters else "")
    )
