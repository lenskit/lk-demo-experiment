"""
Train a recommendation model.

Usage:
  train-model.py [options] ALGO DATASET

Options:
  -m MODULE     import algorithms from MODULE [default: lkdemo.algorithms]
  -o FILE       write trained model to FILE
  -M FILE, --metrics=FILE
                writing training metrics to FILE
  ALGO          name of algorithm to load
  DATASET       name of data set to load
"""

import sys
from docopt import docopt
import pathlib
import importlib
from numcodecs import Blosc
import binpickle
import json
from humanize import naturalsize
try:
    import resource
except ImportError:
    resource = None

from lenskit.util import Stopwatch
from lenskit.algorithms import Recommender
from lkdemo import datasets, log

def main(args):
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
    metrics = {'WallTime': timer.elapsed()}
    _log.info('trained model in %s', timer)
    if resource:
        res = resource.getrusage(resource.RUSAGE_SELF)
        _log.info('%.2fs user, %.2fs system, %.1fMB max RSS', res.ru_utime, res.ru_stime, res.ru_maxrss / 1024)
        metrics.update({
            'UserTime': res.ru_utime,
            'SysTime': res.ru_stime,
            'MaxRSS': res.ru_maxrss,
        })

    if out is None:
        out = f'models/{dsname}-{model}.bpk'
    out = pathlib.Path(out)

    _log.info('writing to %s', out)
    out.parent.mkdir(parents=True, exist_ok=True)
    binpickle.dump(algo, out, codec=Blosc('zstd'))
    stat = out.stat()
    _log.info('model file size: %s', naturalsize(stat.st_size))
    metrics['FileSize'] = stat.st_size


    if '--metrics' in args:
        mf = pathlib.Path(args['--metrics'])
        mf.write_text(json.dumps(metrics))

if __name__ == '__main__':
    _log = log.script(__file__)
    _log.info('arguments received were: %s', sys.argv)
    args = docopt(__doc__)
    main(args)
