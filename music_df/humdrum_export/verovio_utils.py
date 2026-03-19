"""Utilities for safe verovio rendering.

Verovio's C++ runtime can corrupt memory when coexisting with Qt's C++
runtime in the same process, causing intermittent crashes
(``std::length_error``, ``std::bad_alloc``, segfaults). All Verovio work
is therefore done in isolated subprocesses.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_RENDER_TIMEOUT = 60


class VerovioLoadError(Exception):
    """Raised when verovio crashes or times out loading a humdrum string."""


def verovio_render(
    hum: str, *, options: dict | None = None, timeout: int = _RENDER_TIMEOUT
) -> list[str]:
    """Load and render humdrum in a subprocess, returning SVG strings.

    Running Verovio in a separate process avoids C++ runtime conflicts
    with Qt. If the humdrum contains ``!!!filter:`` lines and the render
    crashes, retries without filters.

    Raises:
        VerovioLoadError: If verovio crashes even without filters.
    """
    svgs = _subprocess_render(hum, options=options, timeout=timeout)
    if svgs is not None:
        return svgs

    has_filters = "!!!filter:" in hum
    if has_filters:
        hum_no_filter = "\n".join(
            line for line in hum.split("\n") if not line.startswith("!!!filter:")
        )
        svgs = _subprocess_render(hum_no_filter, options=options, timeout=timeout)
        if svgs is not None:
            logger.warning("Verovio crashed with filters; rendered without them")
            return svgs

    raise VerovioLoadError(
        "Verovio crashed rendering this input"
        + (" even without filters" if has_filters else "")
    )


def _subprocess_render(
    hum: str, *, options: dict | None = None, timeout: int = _RENDER_TIMEOUT
) -> list[str] | None:
    """Render humdrum to SVG strings in a subprocess.

    Returns a list of SVG strings (one per page), or None on failure.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".krn", mode="w", delete=False
    ) as f:
        f.write(hum)
        tmp_path = f.name

    opts_snippet = ""
    if options:
        opts_snippet = f"tk.setOptions({options!r}); "

    script = (
        "import verovio, json, sys; "
        "tk = verovio.toolkit(); "
        f"{opts_snippet}"
        f"tk.loadData(open({tmp_path!r}).read()); "
        "n = tk.getPageCount(); "
        "svgs = [tk.renderToSVG(i) for i in range(1, n + 1)]; "
        "json.dump(svgs, sys.stdout) if n > 0 else sys.exit(1)"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def verovio_safe_load(tk, hum: str, *, options: dict | None = None) -> bool:
    """Load humdrum into verovio, probing in a subprocess first.

    .. deprecated::
        Prefer ``verovio_render`` which does all Verovio work in a subprocess.
        This function still loads data in the main process after probing,
        which can crash when Qt is also loaded.
    """
    svgs = _subprocess_render(hum, options=options)
    if svgs is not None:
        tk.loadData(hum)
        return False

    has_filters = "!!!filter:" in hum
    if has_filters:
        hum_no_filter = "\n".join(
            line for line in hum.split("\n") if not line.startswith("!!!filter:")
        )
        if _subprocess_render(hum_no_filter, options=options) is not None:
            logger.warning("Verovio crashed with filters; loading without them")
            tk.loadData(hum_no_filter)
            return True

    raise VerovioLoadError(
        "Verovio crashed loading this input"
        + (" even without filters" if has_filters else "")
    )
