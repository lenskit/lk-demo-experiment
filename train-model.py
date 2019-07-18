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
try:
  import resource
except ImportError:
  resource = None

from lenskit.util import Stopwatch
from lenskit.algorithms import Recommender
from lkdemo import datasets, log

_log = log.script(__file__)

args = docopt(__doc__)
mod_name = args.get('-m', 'lkdemo.algorithms')
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
algo = Recommender.adapt(algo)
timer = Stopwatch()
algo.fit(ratings)
timer.stop()
_log.info('trained model in %s', timer)
if resource:
    res = resource.getrusage(resource.RUSAGE_SELF)
    _log.info('%.2fs user, %.2fs system, %.1fMB max RSS', res.ru_utime, res.ru_stime, res.ru_maxrss / 1024)

if out is None:
    out = f'models/{dsname}-{model}.pkl.gz'

_log.info('writing to %s', out)
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with gzip.open(out, 'wb') as f:
    pickle.dump(algo, f, 4)
