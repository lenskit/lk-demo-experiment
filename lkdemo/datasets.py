"""
This module defines the data sets that we are going to work with.
"""
import pandas as pd

from lenskit import datasets as ds

ml20m = ds.MovieLens('data/ml-20m')
ml25m = ds.MovieLens('data/ml-25m')
mlsmall = ds.MovieLens('data/ml-latest-small')
ml100k = ds.ML100K('data/ml-100k')
ml1m = ds.ML1M('data/ml-1m')
ml10m = ds.ML10M('data/ml-10M100K')

if hasattr(ds, 'BookCrossing'):
    bx = ds.BookCrossing('data/bx')


def ds_diff(full, subset):
    "Return the difference of two data sets."
    mask = pd.Series(True, index=full.index)
    mask[subset.index] = False
    return full[mask]
