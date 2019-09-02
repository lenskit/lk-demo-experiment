# LensKit Demo Experiment

This repository contains a (work-in-progress) demo experiment for running LensKit experiments on public data sets with current best practices for moderately-sized experiments.

## Layout

This experiment uses [DVC](https://dvc.org) to script the experiment, and is laid out in several subcomponents:

- `lkdemo` is a Python package containing support code (e.g. log configurations) and algorithm definitions
- `data` contains data files and controls
- `models` contains trained LensKit models to run recommendations on individual data sets
- Various Python scripts to run individual pieces of the analysis
- Jupyter notebooks to analyze results

## Setup

This experiment comes with an Anaconda environment file that defines the necessary dependencis.  To set up and activate:

    conda env create -f environment.yml
    conda activate lkdemo

After creating the environment, you just need to `activate`; you can update the environment with `conda env update -f environment.yml`.

## Running

The `dvc` program controls runs of individual steps.  For example, to download the ML-20M data set and train an item-item model, run:

    dvc run models/ml20m-II.dvc
    
## To Do

We have several things to do before this project is ready to be used as documentation

- [ ] Split data for train-test evaluations
- [ ] Provide an example evaluation
- [ ] Provide evaluation analysis notebook
