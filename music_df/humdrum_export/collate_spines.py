import os
import re
import tempfile
import typing as t

import sh


def run_minrhy(files: t.List[str]) -> int:
    assert files
    out = sh.minrhy(*files)  # type:ignore
    m = re.search(r"^all:\t(?P<result>\d+)", out, re.MULTILINE)
    if not m:
        # this occurs if there is only one file; I guess we should just
        #   inspect len(files)
        m = re.search(r"(?P<result>\d+)", out)
    assert m is not None
    out = int(m.group("result"))
    return out


def run_timebase(files: t.List[str], gcd: int) -> t.List[str]:
    tmp_paths = []
    for f in files:
        # with open(f) as inf:
        #     data = inf.read()
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
