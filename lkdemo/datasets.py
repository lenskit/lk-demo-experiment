"""
This module defines the data sets that we are going to work with.
"""

from lenskit.data import Dataset
from lenskit.logging import get_logger
from lenskit.splitting import TTSplit, split_global_time

_log = get_logger(__name__)


def split_fraction(data: Dataset, test_frac: float) -> TTSplit:
    log = _log.bind(name=data.schema.name, test_frac=test_frac)
    logs = data.interaction_table(format="pandas")
    time = logs["timestamp"].quantile(1 - test_frac)
    log.info("splitting at %s", time)
    split = split_global_time(data, time)
    log.info("obtained %d test users", len(split.test))
    return split
