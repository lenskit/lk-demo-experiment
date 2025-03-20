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

```python
from lkdemo.datasets import split_fraction
```

## Load Data

The recommendations are in `runs`, and we will need to reassemble the test data from `test`.

```python tags=["parameters"]
dataset = "ml-100k"
```

```python
output_root = Path("runs")
```

```python
dirs = [fld for fld in output_root.glob(f'{dataset}-*')]
```

```python
recs = ItemListCollection(['model', 'user_id'], index=False)
for fld in dirs:
    for file in fld.glob("recs-*"):
        rec = ItemListCollection.load_parquet(file)
        recs.add_from(rec, model=fld.name.split("-")[-1])
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
        preds.add_from(pred, model=fld.name.split("-")[-1])
```

We need to load the test data so that we have the ground truths for computing accuracy.

```python
data = Dataset.load(f"data/{dataset}")
split = split_fraction(data, 0.2)
test = split.test
```

And identify users in the training set, so we only report metrics over them.

```python
train_users = split.train.user_stats()
train_users = train_users[train_users['rating_count'] > 0]
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

We can reshape the list metrics and plot them, after filtering to only users with at least 1 training rating:

```python
metrics = rec_results.list_metrics()
metrics = metrics.melt(var_name='metric', ignore_index=False).reset_index()
metrics = metrics[metrics['user_id'].isin(train_users.index)]
sns.catplot(metrics, y='model', x='value', col='metric', kind='bar')
plt.show()
```

Let's look at the influence of training ratings on performance, clamping 15+
into a single category — this helps understand perhaps surprising performance
relative to cross-fold evaluations:

```python
tcounts = split.train.user_stats()['rating_count'].copy()
tcounts[tcounts > 15] = 15
metrics = rec_results.list_metrics().reset_index().join(tcounts, on='user_id')
sns.lineplot(metrics, x='rating_count', y='NDCG', hue='model', errorbar='ci')
plt.xlabel('# of Training Ratings')
rc_ticks = np.arange(0, 16, 3)
plt.xticks(rc_ticks, rc_ticks[:-1].tolist() + ['15+'])
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
pred_metrics = pred_metrics[pred_metrics['user_id'].isin(train_users.index)]
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
