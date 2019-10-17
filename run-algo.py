# -*- coding: utf-8 -*-
"""
Usage:
    run-algo.py  [--splits iDIR] [-o oDIR] [-n N] [-m MODULE] ALGO 

Options:  
    --splits iDIR   directory for split train-test pairs [default: splitData]
    -o oDIR         destination directory [default: recs]
    -n N            number of recommendations for a unique user [default: 100]
    -m MODULE       import algorithms from MODULE [default: lkdemo.algorithms]
    ALGO            name of algorithm to load 
"""

from docopt import docopt
from pathlib import Path
from lenskit.algorithms import Recommender, Predictor
from lenskit import batch, util
from lkdemo import log

import importlib
import pandas as pd

_log = log.script(__file__)

args = docopt(__doc__)

mod_name = args.get('-m')
splitsDIR = args.get('--splits')
outDIR = args.get('-o')
n_recs = int(args.get('-n'))
model = args.get('ALGO')

_log.info(f'importing from module {mod_name}')
algorithms = importlib.import_module(mod_name)
algo = getattr(algorithms, model)

Path(outDIR).mkdir(exist_ok=True)
path = Path(splitsDIR)
dest = Path(outDIR)

for file in path.glob("test-*"):
    file_num = (file.stem[-1:])
    test = pd.read_csv(file, sep=',')
    train = pd.read_csv(path / f'train-{file_num}.csv', sep=',')
    users = test.user.unique()
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    _log.info(f'generating recommendations for unique users')  
    recs = batch.recommend(fittable, users, n_recs)
    _log.info(f'writing recommendations to {dest}')
    recs.to_csv(dest / f'recs-{file_num}', index = False)
    
    if isinstance(fittable, Predictor):
        fittable.fit(train)
        preds = batch.predict(fittable, train)
        recs.to_csv(dest / f'pred-{file_num}', index = False)



