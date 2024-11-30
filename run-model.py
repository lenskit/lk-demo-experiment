# -*- coding: utf-8 -*-
"""
Use a model to produce recommendations (and predictions).

Usage:
    run-model.py [options] MODEL

Options:
    -v, --verbose   verbose logging output
    --splits input  directory for split train-test pairs [default: data-split]
    -o output       destination directory [default: output]
    -n N            number of recommendations for a unique user [default: 100]
    -m MODULE       import models from MODULE [default: lkdemo.models]
    -P N, --part=N  only run on partition N
    --no-predict    turn off rating prediction
    --log-file FILE write logs to FILE
    MODEL            name of model to load
"""

import importlib
import logging
from pathlib import Path

import pandas as pd
from docopt import docopt
from lenskit import batch, util
from lenskit.data import ItemListCollection, from_interactions_df
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import TTSplit

from lkdemo import datasets, log

_log = logging.getLogger("run-model")


def main(args):
    mod_name = args.get("-m")
    input = args.get("--splits")
    output = args.get("-o")
    n_recs = int(args.get("-n"))
    model = args.get("MODEL")
    part = args.get("--part", None)

    _log.info(f"importing from module {mod_name}")
    algorithms = importlib.import_module(mod_name)

    model = getattr(algorithms, model)
    pipe = topn_pipeline(model, predicts_ratings=not args["--no-predict"])

    path = Path(input)
    dest = Path(output)
    dest.mkdir(exist_ok=True, parents=True)

    ds_def = getattr(datasets, path.name, None)

    for file in path.glob("test-*.parquet"):
        test = pd.read_parquet(file)
        suffix = file.name[5:]
        _log.debug("checking file %s against %s", file.stem[5:], part)
        if part is not None and file.stem[5:] != part:
            _log.info("part %s not wanted, skipping", suffix)
            continue

        train_file = path / f"train-{suffix}"
        timer = util.Stopwatch()

        test = ItemListCollection.from_df(test)

        if train_file.exists():
            _log.info("[%s] loading training data from %s", timer, train_file)
            train = pd.read_parquet(path / f"train-{suffix}")
            train = from_interactions_df(train)
            split = TTSplit(train, test)
        elif ds_def is not None:
            _log.info(
                "[%s] extracting training data from data set %s", timer, path.name
            )
            split = TTSplit.from_src_and_test(ds_def(), test)
        else:
            _log.error("could not find training data for %s", file.name)
            continue

        _log.info("[%s] Fitting the model", timer)
        copy = pipe.clone()
        copy.train()

        try:
            _log.info("[%s] generating recommendations for unique users", timer)
            recs = batch.recommend(pipe, split.test.keys(), n_recs)
            _log.info("[%s] writing recommendations to %s", timer, dest)
            recs.to_df().to_parquet(
                dest / f"recs-{suffix}", index=False, compression="zstd"
            )

            if not args["--no-predict"]:
                _log.info("[%s] generating predictions for user-item", timer)
                preds = batch.predict(pipe, split.test)
                preds.to_parquet(
                    dest / f"pred-{suffix}", index=False, compression="zstd"
                )
        finally:
            model.close()


if __name__ == "__main__":
    args = docopt(__doc__)
    _log = log.script(__file__, debug=args["--verbose"], log_file=args["--log-file"])
    main(args)
