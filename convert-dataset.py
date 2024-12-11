"""
Convert a dataset into Parquet for efficient loading.

Usage:
    convert-dataset.py [-v] [-o OUT] --movielens INPUT

Options:
    --movielens
            Convert MovieLens data.
    -v, --verbose
            enable verbose logging
    -o OUT, --output=OUT
            name of the output file (derived from input by default).
    INPUT
            name of the input file to convert.
"""

import sys
from pathlib import Path

import structlog
from docopt import docopt
from lenskit.data import load_movielens
from lenskit.logging import LoggingConfig

_log = structlog.stdlib.get_logger("convert-dataset")


def main(args):
    if args["--movielens"]:
        import_movielens(args)
    else:
        _log.error("no data set type specified")
        sys.exit(2)


def import_movielens(args):
    infile = Path(args.get("INPUT"))
    log = _log.bind(src=str(infile))

    log.info("loading MovieLens data")
    data = load_movielens(infile)
    log.info("loaded %d ratings", data.interaction_count)

    out = args.get("--output", None)
    if out is None:
        out = infile.with_suffix(".parquet")

    log.info("writing Parquet output", dst=str(out))
    data.interaction_log("pandas", original_ids=True).to_parquet(
        out, compression="zstd"
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    lcfg = LoggingConfig()
    lcfg.set_verbose(args["--verbose"])
    lcfg.apply()
    main(args)
