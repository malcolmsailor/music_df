"""Utilities for safe verovio rendering."""

import subprocess
import sys
import tempfile
from pathlib import Path


def verovio_safe_load(tk, hum: str) -> bool:
    """Load humdrum into verovio, falling back to no filters if autobeam crashes.

    Verovio's autobeam filter can crash (SIGABRT, exit 134) on complex
    durations like triple-dotted eighths from mid-measure crops. This
    function probes in a subprocess first; if the subprocess crashes,
    it strips all !!!filter: lines and loads without them.

    Args:
        tk: A verovio.toolkit() instance.
        hum: Humdrum string to load.

    Returns:
        True if filters were stripped (fallback was used), False otherwise.
    """
    if "!!!filter:" not in hum:
        tk.loadData(hum)
        return False

    with tempfile.NamedTemporaryFile(
        suffix=".krn", mode="w", delete=False
    ) as f:
        f.write(hum)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import verovio, sys; "
                "tk = verovio.toolkit(); "
                f"tk.loadData(open({tmp_path!r}).read()); "
                "sys.exit(0 if tk.getPageCount() > 0 else 1)",
            ],
            capture_output=True,
            timeout=30,
        )
        autobeam_ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        autobeam_ok = False
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if autobeam_ok:
        tk.loadData(hum)
        return False
    else:
        hum_no_filter = "\n".join(
            line for line in hum.split("\n") if not line.startswith("!!!filter:")
        )
        tk.loadData(hum_no_filter)
        return True
