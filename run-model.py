# -*- coding: utf-8 -*-
"""
Use a model to produce recommendations (and predictions).  It uses a global
temporal split determined by fraction of the test data.

Usage:
    run-model.py [options] MODEL

Options:
    -v, --verbose   verbose logging output
    -d DIR, --data=DIR
            directory of the dataset (in LensKit native format)
    --test-size=F   the test set size (as a fraction) [default: 0.2]
    -o output       destination directory [default: output]
    -n N            number of recommendations for a unique user [default: 100]
    -m MODULE       import models from MODULE [default: lkdemo.models]
    --no-predict    turn off rating prediction
    --log-file FILE write logs to FILE
    MODEL            name of model to load
"""

import importlib
import logging
from pathlib import Path

from docopt import docopt
from lenskit.batch import BatchPipelineRunner
from lenskit.data import Dataset
from lenskit.logging.config import LoggingConfig
from lenskit.pipeline import topn_pipeline

from lkdemo.datasets import split_fraction

_log = logging.getLogger("run-model")


def main(args):
    mod_name = args.get("-m")
    output = args.get("-o")
    n_recs = int(args.get("-n"))
    model = args.get("MODEL")

    data_path = args.get("--data")
    data = Dataset.load(data_path)
    quant = float(args["--test-size"])

    split = split_fraction(data, quant)

    _log.info(f"importing from module {mod_name}")
    algorithms = importlib.import_module(mod_name)

    model = getattr(algorithms, model)
    pipe = topn_pipeline(model, predicts_ratings=not args["--no-predict"])

    dest = Path(output)
    dest.mkdir(exist_ok=True, parents=True)

    _log.info("training the pipeline")
    pipe.train(split.train)

    _log.info(
        "generating recommendations for %d unique users",
        len(split.test),
    )
    runner = BatchPipelineRunner()
    runner.recommend(n=n_recs)
    if not args["--no-predict"]:
        runner.predict()

    result = runner.run(pipe, split.test)
    recs = result.output("recommendations")
    _log.info("writing recommendations to %s", dest)
    recs.save_parquet(dest / "recs-eval.parquet", compression="zstd")

    if not args["--no-predict"]:
        preds = result.output("predictions")
        _log.info("writing predictions to %s", dest)
        preds.save_parquet(dest / "pred-eval.parquet", compression="zstd")


if __name__ == "__main__":
    args = docopt(__doc__)
    lcfg = LoggingConfig()
    if args["--verbose"]:
        lcfg.set_verbose()
    if args["--log-file"]:
        lcfg.log_file(args["--log-file"], logging.DEBUG)
    lcfg.apply()
    main(args)
