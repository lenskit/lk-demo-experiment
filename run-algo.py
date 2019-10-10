# -*- coding: utf-8 -*-
"""
Usage:
    run-algo.py  [--splits iDIR] [-o oDIR] ALGO

Options:  
    --splits iDIR   number of cross-folds [default: splitData]
    -o oDIR         destination directory [default: recs]
    ALGO            name of data set to load  
"""

from docopt import docopt
from pathlib import Path
from lenskit.algorithms import Recommender
from lenskit import batch
from lkdemo import log

import gzip
import pandas as pd
import pickle

_log = log.script(__file__)

args = docopt(__doc__)

splitsDIR = args.get('--splits')
outDIR = args.get('-o')
algo = args.get('ALGO')

path = Path(splitsDIR)

n_users = 100
_log.info(f'generating recommendations for {n_users} unique users')
all_recs = []
test_data = []
for i in range(len(list(path.iterdir()))//2):
    train = pd.read_csv(f'train-{i+1}.csv', sep=',')
    test = pd.read_csv(f'test-{i}.csv', sep=',')
    test_data.append(test)
    
    algo = Recommender.adapt(algo)
    algo.fit(train)
    users = test.user.unique()
    recs = batch.recommend(algo, users, n_users)
    recs['Algorithm'] = algo.__class__.__name__    
    all_recs.append(recs)
    all_recs = pd.concat(all_recs, ignore_index=True)

_log.info('writing to %s', outDIR)
Path(outDIR).parent.mkdir(parents=True, exist_ok=True)
with gzip.open(outDIR, 'wb') as f:
    pickle.dump(algo, f, 4)

