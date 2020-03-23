# -*- coding: utf-8 -*-
"""
Usage:
    run-algo.py  [--splits input] [-o output] [-n N] [-m MODULE] ALGO 

Options:  
    --splits input  directory for split train-test pairs [default: data-split]
    -o output       destination directory [default: output]
    -n N            number of recommendations for a unique user [default: 100]
    -m MODULE       import algorithms from MODULE [default: lkdemo.algorithms]
    ALGO            name of algorithm to load 
"""

import os
import multiprocessing
from docopt import docopt
from pathlib import Path
from lenskit.algorithms import Recommender, Predictor
from lenskit import batch, util
from lkdemo import log, datasets

import importlib
import pandas as pd

_log = log.script(__file__)

args = docopt(__doc__)

mod_name = args.get('-m')
input = args.get('--splits')
output = args.get('-o')
n_recs = int(args.get('-n'))
model = args.get('ALGO')

ncpus = os.environ.get('LK_NUM_PROCS', None)
if ncpus is None:
    ncpus = max(multiprocessing.cpu_count() // 2, 1)
else:
    ncpus = int(ncpus)


_log.info(f'importing from module {mod_name}')
algorithms = importlib.import_module(mod_name)

algo = getattr(algorithms, model)

path = Path(input)
dest = Path(output)
dest.mkdir(exist_ok=True , parents=True)

ds_def = getattr(datasets, path.name, None)

for file in path.glob("test-*"):
    test = pd.read_csv(file, sep=',')
    suffix = file.name[5:]
    train_file = path / f'train-{suffix}'
    
    if 'index' in test.columns:
        _log.info('setting test index')
        test = test.set_index('index')
    else:
        _log.warn('no index column found in %s', file.name)

    if train_file.exists():
        _log.info('loading training data from %s', train_file)
        train = pd.read_csv(path / f'train-{suffix}', sep=',')
    elif ds_def is not None:
        _log.info('extracting training data from data set %s', path.name)
        train = datasets.ds_diff(ds_def.ratings, test)
        train.reset_index(drop=True, inplace=True)
    else:
        _log.error('could not find training data for %s', file.name)
        continue

    _log.info('Fitting the model')
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    
    _log.info(f'generating recommendations for unique users')  
    users = test.user.unique()
    recs = batch.recommend(fittable, users, n_recs)
    _log.info(f'writing recommendations to {dest}')
    suffix = model + suffix
    recs.to_csv(dest / f'recs-{suffix}', index = False)
    
    if isinstance(fittable, Predictor):
        _log.info(f'generating predictions for user-item') 
        preds = batch.predict(fittable, test)
        preds.to_csv(dest / f'pred-{suffix}', index = False)
