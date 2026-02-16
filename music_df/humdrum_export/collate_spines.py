import math
import os
import re
import tempfile
import typing as t

try:
    import sh
except ImportError as e:
    raise ImportError(
        "sh is required for humdrum export features. "
        "Install with: pip install music_df[humdrum]"
    ) from e


def run_minrhy(files: t.List[str]) -> int:
    assert files
    try:
        out = sh.minrhy(*files)  # type:ignore
    except sh.ErrorReturnCode as e:
        raise RuntimeError(
            f"minrhy failed on {len(files)} file(s). "
            f"This may be caused by malformed kern files "
            f"(e.g. from a mid-measure crop).\n"
            f"stderr: {e.stderr.decode() if e.stderr else '(empty)'}"
        ) from e
    m = re.search(r"^all:\t(?P<result>\d+)", out, re.MULTILINE)
    if not m:
        # this occurs if there is only one file; I guess we should just
        #   inspect len(files)
        m = re.search(r"(?P<result>\d+)", out)
    assert m is not None
    out = int(m.group("result"))
    return out


def _clean_up(tmp_paths):
    for f in tmp_paths:
        os.remove(f)


def _collect_recips(files: t.List[str]) -> int | None:
    """Return the GCD of all % recip denominators across files, or None."""
    all_recips = []
    for f in files:
        with open(f) as inf:
            data = inf.read()
        all_recips.extend(int(r) for r in re.findall(r"\d%(\d+)", data))
    return math.gcd(*all_recips) if all_recips else None


def _rscale_files(files: t.List[str], factor: str) -> t.List[str]:
    """rscale each file, returning paths to new temp files."""
    tmp_paths = []
    for f in files:
        _, tmp_path = tempfile.mkstemp(suffix=".krn")
        result = sh.rscale("-f", factor, f)  # type:ignore
        with open(tmp_path, "w") as outf:
            outf.write(str(result))
        tmp_paths.append(tmp_path)
    return tmp_paths


def collate_spines(files: t.List[str]):
    # The humdrum toolkit (minrhy, timebase) doesn't understand the "%"
    # notation for irrational durations (e.g., 8%9 for a 9/8 rest). We
    # rscale ALL files up front to remove %, run the standard pipeline,
    # then rscale the result back. The rscale must be global (applied to
    # all files, not just those containing %) so that all files end up on
    # the same timebase grid.
    global_recip = _collect_recips(files)

    if global_recip is not None:
        scaled_files = _rscale_files(files, f"1/{global_recip}")
        work_files = scaled_files
    else:
        scaled_files = None
        work_files = files

    try:
        gcd = run_minrhy(work_files)
        tb_paths = []
        for f in work_files:
            result = sh.timebase("-t", str(gcd), f)  # type:ignore
            _, tmp_path = tempfile.mkstemp(suffix=".krn")
            with open(tmp_path, "w") as outf:
                outf.write(str(result))
            tb_paths.append(tmp_path)

        # rid strips null records (lines consisting only of ".")
        result = sh.rid("-d", _in=sh.assemble(*tb_paths))  # type:ignore
        _clean_up(tb_paths)
    finally:
        if scaled_files is not None:
            _clean_up(scaled_files)

    if global_recip is not None:
        _, tmp_path = tempfile.mkstemp(suffix=".krn")
        with open(tmp_path, "w") as outf:
            outf.write(str(result))
        result = sh.rscale("-f", str(global_recip), tmp_path)  # type:ignore
        os.remove(tmp_path)

    return str(result)
