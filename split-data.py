# -*- coding: utf-8 -*-
"""
Usage:
    split-data.py [-v] [-p partitions] [-o output] DATASET

Options:
    -v, --verbose   enable verbose logging
    -p PARTS        number of cross-folds [default: 5]
    -o OUT          destination directory [default: data-split]
    DATASET         name of data set to load
"""

import logging
from pathlib import Path

from docopt import docopt
from lenskit.logging import LoggingConfig
from lenskit.splitting import SampleN, crossfold_users

from lkdemo import datasets

_log = logging.getLogger("split-data")


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
        tp.test.to_df().to_parquet(path / f"test-{i}.parquet")


if __name__ == "__main__":
    args = docopt(__doc__)
    lcfg = LoggingConfig()
    if args["--verbose"]:
        lcfg.set_verbose()
    lcfg.apply()
    main(args)
