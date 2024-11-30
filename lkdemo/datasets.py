"""
This module defines the data sets that we are going to work with.
"""

from lenskit.data import Dataset, load_movielens


def ml20m() -> Dataset:
    return load_movielens("data/ml-20m.zip")


def ml25m() -> Dataset:
    return load_movielens("data/ml-25m.zip")


def mlsmall() -> Dataset:
    return load_movielens("data/ml-latest-small.zip")


def ml100k() -> Dataset:
    return load_movielens("data/ml-100k.zip")


def ml1m() -> Dataset:
    return load_movielens("data/ml-1m.zip")


def ml10m() -> Dataset:
    return load_movielens("data/ml-10M100K.zip")
