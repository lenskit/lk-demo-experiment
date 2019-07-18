"""
Generate recommendations.

Usage:
  recommend.py [-n N] [-d DS] MODEL USER ...

Options:
  -n N      the number of recommendations to produce [default: 10]
  -d DS     the data set, for reading item information
  MODEL     the name of the model file
  USER      the user IDs to recommend for
"""

from docopt import docopt
import gzip
import pickle

from lkdemo import datasets, log
from lenskit.util import Stopwatch

_log = log.script(__file__)

args = docopt(__doc__)
n = int(args['-n'])

if args['-d']:
    _log.info('using data %s', args['-d'])
    data = getattr(datasets, args['-d'])
    items = data.movies
else:
    data = None
    items = None

_log.info('reading from %s', args['MODEL'])
with gzip.open(args['MODEL'], 'rb') as f:
    algo = pickle.load(f)


for u in args['USER']:
    u = int(u)
    timer = Stopwatch()
    _log.info('getting %d recs for user %d', n, u)
    recs = algo.recommend(u, n)
    if items is not None:
        recs = recs.join(items, how='left', on='item')
    print('recommendations for', u)
    print(recs)
    _log.info('completed recommendations in %s', timer)
