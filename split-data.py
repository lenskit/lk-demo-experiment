# -*- coding: utf-8 -*-

"""
Usage:
    split-data.py [-p partitions] [-o output] DATASET

Options:  
    -p partitions     number of cross-folds [default: 5]
    -o output         destination directory [default: data-split]
    DATASET           name of data set to load  
"""

from docopt import docopt
from lkdemo import datasets, log
from pathlib import Path

import lenskit.crossfold as xf

_log = log.script(__file__)

args = docopt(__doc__)

dsname = args.get('DATASET')
partitions = int(args.get('-p'))
output = args.get('-o')

_log.info('locating data set %s', dsname)
data = getattr(datasets, dsname)

_log.info('loading ratings')
ratings = data.ratings

path = Path(output)
path.mkdir(exist_ok=True, parents=True)

_log.info('writing to %s', path)
testRowsPerUsers = 5
for i, tp in enumerate(xf.partition_users(ratings, partitions, xf.SampleN(testRowsPerUsers)),1):
    tp.train.to_csv(path / f'train-{i}.csv.gz', index=False)
    tp.test.to_csv(path / f'test-{i}.csv.gz' , index=False)
    