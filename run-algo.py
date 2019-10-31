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

path = Path(splitsDIR)
dest = Path(outDIR)
dest.mkdir(exist_ok=True)
#test-*.csv
for file in path.glob("test-*"):
    
    #check .csv (if not csv then an error is thrown out)
    try:
        test = pd.read_csv(file, sep=',')
        suffix = file.name[5:]
        train = pd.read_csv(path / f'train-{suffix}', sep=',')
        users = test.user.unique()
        
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train)
        
        _log.info(f'generating recommendations for unique users')  
        recs = batch.recommend(fittable, users, n_recs)
        recs["Algorithm"] = model
        _log.info(f'writing recommendations to {dest}')
        suffix = model + suffix
        recs.to_csv(dest / f'recs-{suffix}', index = False)
        
        if isinstance(fittable, Predictor):
            _log.info(f'generating predictions for user-item') 
            preds = batch.predict(fittable, test)
            preds["Algorithm"] = model
            preds.to_csv(dest / f'pred-{suffix}', index = False)
    except FileNotFoundError:
        _log.error(f'{file} is not a csv file')
    


