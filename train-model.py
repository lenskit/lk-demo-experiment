"""
Train a recommendation model.

Usage:
  train-model.py [-m MODULE] [-o FILE] ALGO DATASET

Options:
  -m MODULE     import algorithms from MODULE
  -o FILE       write trained model to FILE
  ALGO          name of algorithm to load
  DATASET       name of data set to load
"""

from docopt import docopt
import pathlib
import importlib
import gzip
import pickle

from lkdemo import datasets, log

_log = log.script(__file__)

args = docopt(__doc__)
mod_name = args.get('MODULE', 'lkdemo.algorithms')
out = args.get('FILE', None)
model = args.get('ALGO')
dsname = args.get('DATASET')

_log.info('importing from module %s', mod_name)
algorithms = importlib.import_module(mod_name)

_log.info('locating model %s', model)
algo = getattr(algorithms, model)
_log.info('locating data set %s', dsname)
data = getattr(datasets, dsname)

_log.info('loading ratings')
ratings = data.ratings
_log.info('training model')
algo.fit(ratings)

if out is None:
    out = f'models/{dsname}-{model}.pkl.gz'

_log.info('writing to %s', out)
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with gzip.open(out, 'wb') as f:
    pickle.dump(algo, f, 4)