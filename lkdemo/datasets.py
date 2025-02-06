"""
This module defines the data sets that we are going to work with.
"""

from lenskit.data import Dataset, load_movielens
from lenskit.splitting import TTSplit, split_global_time


def split_fraction(data: Dataset, test_frac: float) -> TTSplit:
    logs = data.interaction_table(format="pandas")
    time = logs["timestamp"].quantile(1 - test_frac)
    return split_global_time(data, time)
