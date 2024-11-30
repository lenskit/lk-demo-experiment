# -*- coding: utf-8 -*-
"""
Usage:
    split-data.py [-p partitions] [-o output] DATASET

Options:
    -p partitions     number of cross-folds [default: 5]
    -o output         destination directory [default: data-split]
    DATASET           name of data set to load
"""

from pathlib import Path

from docopt import docopt
from lenskit.splitting import SampleN, crossfold_users

from lkdemo import datasets, log


def main(args):
    dsname = args.get("DATASET")
    partitions = int(args.get("-p"))
    output = args.get("-o")

    _log.info("locating data set %s", dsname)
    data = getattr(datasets, dsname)

    _log.info("loading ratings")
    ratings = data()

    path = Path(output)
    path.mkdir(exist_ok=True, parents=True)

    _log.info("writing to %s", path)
    testRowsPerUsers = 5
    for i, tp in enumerate(
        crossfold_users(ratings, partitions, SampleN(testRowsPerUsers)), 1
    ):
        _log.info("writing test set %d", i)
        tp.test.to_parquet(path / f"test-{i}.parquet")


if __name__ == "__main__":
    _log = log.script(__file__)
    args = docopt(__doc__)
    main(args)
