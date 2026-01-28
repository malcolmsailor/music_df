import glob
from multiprocessing import Pool
import os
import sys
from dataclasses import dataclass

try:
    from omegaconf import OmegaConf
except ImportError as e:
    raise ImportError(
        "omegaconf is required for this script. "
        "Install with: pip install music_df[scripts]"
    ) from e
import traceback, pdb, sys

import pandas as pd
from tqdm import tqdm

from music_df.read_csv import read_csv


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    input_dir: str
    num_workers: int = 16


def get_csv_files(folder_path):
    # Use glob.glob with recursive=True to get all matching files
    return glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)


def do_file(path):
    df = read_csv(path)
    assert df is not None, f"couldn't read {path}"
    if "type" not in df.columns:
        return
    return (df.type == "note").sum()


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    csv_files = get_csv_files(config.input_dir)
    # result = [do_file(x) for x in csv_files]
    with Pool(config.num_workers) as pool:
        result = list(
            tqdm(pool.imap_unordered(do_file, csv_files), total=len(csv_files))
        )

    result_df = pd.DataFrame({"counts": [x for x in result if x is not None]})
    print(f"Total notes: {result_df.counts.sum()}")
    print(result_df.describe())


if __name__ == "__main__":
    main()
