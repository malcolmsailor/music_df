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


# def run_timebase(files: t.List[str], gcd: int) -> t.List[str]:
#     tmp_paths = []
#     for f in files:
#         # with open(f) as inf:
#         #     data = inf.read()
#         result = sh.timebase("-t", str(gcd), f)  # type:ignore
#         _, tmp_path = tempfile.mkstemp(suffix=".krn")
#         with open(tmp_path, "w") as outf:
#             outf.write(result)
#         tmp_paths.append(tmp_path)
#     return tmp_paths


def run_timebase(files: t.List[str], gcd: int) -> t.List[str]:
    tmp_paths = []
    for f in files:
        with open(f) as inf:
            data = inf.read()
        recips = re.findall(r"\d%(\d+)", data)
        if recips:
            # because `timebase`, being from the original humdrum toolkit, doesn't
            #   understand the "%" notation (e.g., 2%3), we first use rscale to scale up
            #   the file to the gcd of the recips (ensuring there will be no %), and
            #   then afterwards scale it back
            recip = math.gcd(*[int(r) for r in recips])

            _, tmp_path = tempfile.mkstemp(suffix=".krn")
            rscale1_result = sh.rscale("-f", f"1/{recip}", f)  # type:ignore
            with open(tmp_path, "w") as outf:
                outf.write(rscale1_result)
            timebase_result = sh.timebase(  # type:ignore
                "-t", str(gcd), tmp_path
            )
            with open(tmp_path, "w") as outf:
                outf.write(timebase_result)
            rscale2_result = sh.rscale("-f", f"{recip}", tmp_path)  # type:ignore
            with open(tmp_path, "w") as outf:
                outf.write(rscale2_result)
            tmp_paths.append(tmp_path)
        else:
            result = sh.timebase("-t", str(gcd), f)  # type:ignore
            _, tmp_path = tempfile.mkstemp(suffix=".krn")
            with open(tmp_path, "w") as outf:
                outf.write(result)
            tmp_paths.append(tmp_path)
    return tmp_paths


def _clean_up(tmp_paths):
    for f in tmp_paths:
        os.remove(f)


def collate_spines(files: t.List[str]):
    gcd = run_minrhy(files)
    tmp_paths = run_timebase(files, gcd)
    # rid strips null records (lines consisting only of ".")
    result = sh.rid("-d", _in=sh.assemble(*tmp_paths))  # type:ignore
    _clean_up(tmp_paths)
    return result
