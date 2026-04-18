#!/usr/bin/env python3
"""
Use a pipeline to produce recommendations (and predictions).  It uses a global
temporal split determined by fraction of the test data.

Usage:
    run-pipeline.py [options] PIPE

Options:
    -v, --verbose   verbose logging output
    -d DIR, --data=DIR
            directory of the dataset (in LensKit native format)
    -o output       destination directory [default: output]
    -n N            number of recommendations for a unique user [default: 100]
    --log-file FILE write logs to FILE
    PIPE            pipeline configuration file to load
"""

import logging
from pathlib import Path

from docopt import docopt
from lenskit import Pipeline
from lenskit.batch import BatchPipelineRunner
from lenskit.config import configure
from lenskit.data import Dataset, ItemListCollection
from lenskit.logging import LoggingConfig

_log = logging.getLogger("run-model")


def main(args):
    configure()
    output = args.get("-o")
    n_recs = int(args.get("-n"))

    data_path = Path(args.get("--data"))

    pipe_file = args["PIPE"]
    _log.info("loading pipeline from %s", pipe_file)
    pipeline = Pipeline.load_config(pipe_file)

    predicts_ratings = pipeline.node("rating-predictor", missing=None) is not None

    _log.info("loading data from %s", data_path)
    train = Dataset.load(data_path / "train")
    test = ItemListCollection.load_parquet(data_path / "test.parquet")

    dest = Path(output)
    dest.mkdir(exist_ok=True, parents=True)

    _log.info("training the pipeline")
    pipeline.train(train)

    _log.info(
        "generating recommendations for %d unique users",
        len(test),
    )
    runner = BatchPipelineRunner()
    runner.recommend(n=n_recs)
    if predicts_ratings:
        _log.info("pipeline %s can predict ratings")
        runner.predict()

    result = runner.run(pipeline, test)
    recs = result.output("recommendations")
    _log.info("writing recommendations to %s", dest)
    recs.save_parquet(dest / "recs-eval.parquet", compression="zstd")

    if predicts_ratings:
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
