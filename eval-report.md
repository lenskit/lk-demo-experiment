---
jupyter:
  jupytext:
    formats: ipynb,md
    notebook_metadata_filter: split_at_heading
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: .venv
    language: python
    name: python3
  split_at_heading: true
---

# Evaluation Analysis for Recommender Output


In this section we will analyze and compare the generated recommendations and predictions from a predefined list of algorithms with the goal of assessing the performance of each algorithm with respect to a metric. In other words, we would rank the algorithms for each metric considered with respect to performance.


## Setup


Below are the list of packages required to successfully run the analysis. They are divided into partitions to signify their specific task.<br>
We need the pathlib package for working with files and folders

```python
from pathlib import Path
import json
```

Load libraries for analysis and visualization:

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


Import the LensKit metrics for analysis:

```python
from lenskit.data import Dataset, ItemListCollection
from lenskit.metrics import RunAnalysis, RMSE, NDCG, RecipRank, RBP
```

## Load Data

The recommendations are in `runs`, and we will need to reassemble the test data from `test`.

```python tags=["parameters"]
dataset = "ml-100k"
```

```python
output_root = Path("runs")
run_dir = output_root / dataset
```

```python
dirs = [rd for rd in run_dir.iterdir() if rd.is_dir()]
```

```python
recs = ItemListCollection(['model', 'user_id'], index=False)
for fld in dirs:
    for file in fld.glob("recs-*"):
        rec = ItemListCollection.load_parquet(file)
        recs.add_from(rec, model=fld.name)
```

```python
rec_algos = sorted(set(a for (a, _u) in recs.keys()))
rec_algos
```

```python
preds = ItemListCollection(['model', 'user_id'], index=False)
for fld in dirs:
    for file in fld.glob("pred-*"):
        pred = ItemListCollection.load_parquet(file)
        preds.add_from(pred, model=fld.name)
```

We need to load the test data so that we have the ground truths for computing accuracy.

```python
test = ItemListCollection.load_parquet(f"data-split/{dataset}/test.parquet")
```

## Top-N Metrics

`RunListAnalysis` computes metrics for recommendation results and takes care of
matching recommendations and ground truth.

```python
ra = RunAnalysis()

ra.add_metric(NDCG())
ra.add_metric(RecipRank())
ra.add_metric(RBP())

rec_results = ra.compute(recs, test)
rec_results.list_summary('model')
```

We can reshape the list metrics and plot them:

```python
metrics = rec_results.list_metrics()
metrics = metrics.melt(var_name='metric', ignore_index=False).reset_index()
sns.catplot(metrics, y='model', x='value', col='metric', kind='bar')
plt.show()
```

## Prediction RMSE

We will also look at the prediction RMSE.

```python
pa = RunAnalysis()

pa.add_metric(RMSE(missing_scores='ignore', missing_truth='ignore'))

pred_results = pa.compute(preds, test)
pred_results.list_summary('model')
```

```python
pred_metrics = pred_results.list_metrics().reset_index()
sns.catplot(pred_metrics, x='model', y='RMSE', kind='bar')
plt.show()
```

## Save Metrics

We'll now save the metrics to a file.

```python
rlsum = rec_results.list_summary('model')['mean'].unstack()
rlsum
```

```python
rlsum.to_json(f'eval-metrics.{dataset}.json', orient='index')
```
