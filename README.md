# LensKit Demo Experiment

This repository contains a demo experiment for running LensKit experiments on
public data sets with current best practices for moderately-sized experiments.

## Layout

This experiment uses [DVC](https://dvc.org) to script the experiment, and is
laid out in several subcomponents:

- `lkdemo` is a Python package containing support code (e.g. log configurations)
  and algorithm definitions.  Two files are of particular interest:

    - `lkdemo/algorithms.py` defines the different algorithms we can train with
      sensible default configurations.
    - `lkdemo/datasets.py` defines the different data sets, so that any
      supported data set can be loaded into the format [LensKit expects][lkdata]
      in a uniform fashion.

- `data` contains data files and controls.

- `data-split` contains cross-validation splits, produced by `split-data.py`.
  These splits only contain the test files, to save disk space - the train files
  can be obtained with `lkdemo.datasets.ds_diff`, as seen in `run-algo.py`.

- `runs` contains the results of running LensKit train/test runs.

- Various Python scripts to run individual pieces of the analysis.  They use
  `docopt` for parsing their arguments and thus have comprehensive usage docs
  in their docstrings.

- Jupyter notebooks to analyze results.  These are parameterized and run with
  [Papermill][] to analyze different data sets with the same notebook.

[Papermill]: https://papermill.readthedocs.io/en/latest/
[lkdata]: https://lkpy.lenskit.org/en/stable/datasets.html

## Setup

This experiment comes with an Anaconda environment file that defines the
necessary dependencies.  To set up and activate:

    conda env create -f environment.yml
    conda activate lk-demo

After creating the environment, you just need to `activate`; you can update the
environment with `conda env update -f environment.yml`.

## Running

The `dvc` program controls runs of individual steps, including downloading data.
For example, to download the ML-20M data set and recommend with ALS, run:

    dvc repro runs/dvc.yaml:ml20m@ALS

To re-run the whole experiment:

    dvc repro

To reproduce results on one data set:

    dvc repro eval-report-ml100k

## Extending

The various `dvc.yaml` files control the run.  Look at them to modify and extend!

You will probably want to consult the [DVC user guide][dvc].

[dvc]: https://dvc.org/doc/user-guide/
