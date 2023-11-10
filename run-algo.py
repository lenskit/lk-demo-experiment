# -*- coding: utf-8 -*-
"""
Run an algorithm to produce predictions and recommendations.

Usage:
    run-algo.py [options] ALGO

Options:
    -v, --verbose   verbose logging output
    --splits input  directory for split train-test pairs [default: data-split]
    -o output       destination directory [default: output]
    -n N            number of recommendations for a unique user [default: 100]
    -m MODULE       import algorithms from MODULE [default: lkdemo.algorithms]
    -P N, --part=N  only run on partition N
    --no-predict    turn off rating prediction
    --log-file FILE write logs to FILE
    ALGO            name of algorithm to load
"""

import os
import multiprocessing
from docopt import docopt
from pathlib import Path
from lenskit.algorithms import Recommender, Predictor
from lenskit import batch, util
from seedbank import init_file

from lkdemo import log, datasets

import importlib
import pandas as pd


def main(args):
    mod_name = args.get('-m')
    input = args.get('--splits')
    output = args.get('-o')
    n_recs = int(args.get('-n'))
    model = args.get('ALGO')
    part = args.get('--part', None)

    init_file('params.yaml', 'run-algo', model)

    _log.info(f'importing from module {mod_name}')
    algorithms = importlib.import_module(mod_name)

    algo = getattr(algorithms, model)
    algo = Recommender.adapt(algo)

    path = Path(input)
    dest = Path(output)
    dest.mkdir(exist_ok=True , parents=True)

    ds_def = getattr(datasets, path.name, None)

    for file in path.glob("test-*.parquet"):
        test = pd.read_parquet(file)
        suffix = file.name[5:]
        _log.debug('checking file %s against %s', file.stem[5:], part)
        if part is not None and file.stem[5:] != part:
            _log.info('part %s not wanted, skipping', suffix)
            continue

        train_file = path / f'train-{suffix}'
        timer = util.Stopwatch()

        if 'index' in test.columns:
            _log.info('setting test index')
            test = test.set_index('index')
        else:
            _log.warning('no index column found in %s', file.name)

        if train_file.exists():
            _log.info('[%s] loading training data from %s', timer, train_file)
            train = pd.read_csv(path / f'train-{suffix}', sep=',')
        elif ds_def is not None:
            _log.info('[%s] extracting training data from data set %s', timer, path.name)
            train = datasets.ds_diff(ds_def.ratings, test)
            train.reset_index(drop=True, inplace=True)
        else:
            _log.error('could not find training data for %s', file.name)
            continue

        _log.info('[%s] Fitting the model', timer)
        model = batch.train_isolated(algo, train)

        try:
            _log.info('[%s] generating recommendations for unique users', timer)
            users = test.user.unique()
            recs = batch.recommend(model, users, n_recs)
            _log.info('[%s] writing recommendations to %s', timer, dest)
            recs.to_csv(dest / f'recs-{suffix}', index=False)

            if isinstance(algo, Predictor) and not args['--no-predict']:
                _log.info('[%s] generating predictions for user-item', timer)
                preds = batch.predict(model, test)
                preds.to_csv(dest / f'pred-{suffix}', index=False)
        finally:
            model.close()

if __name__ == '__main__':
    args = docopt(__doc__)
    _log = log.script(__file__, debug=args['--verbose'], log_file=args['--log-file'])
    main(args)
