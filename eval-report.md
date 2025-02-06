---
jupyter:
  jupytext:
    formats: ipynb,md
    notebook_metadata_filter: split_at_heading
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
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

We would use the pandas for analyzing and manipulating our data while seaborn and matplotlib are used for data visualization. statsmodels.graphics.gofplots and scipy.stats.shapiro are used for normality check. Scipy.stats.friedmanchisquare is a non-parametric test used to determine the statistical significance in metric results and the wilcoxon test is used for pairwise comparison of sample data.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
```


Import the LensKit metrics for analysis:

```python
from lenskit.data import ItemListCollection, UserIDKey
from lenskit.metrics import RunAnalysis, RMSE, NDCG, RecipRank, RBP
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

We need to load the test data so that we have the ground truths for computing accuracy

```python
split_root = Path("data-split")
split_dir = split_root / dataset
```

```python
test = ItemListCollection(UserIDKey)
for file in split_dir.glob("test-*.parquet"):
    df = pd.read_parquet(file)
    part = ItemListCollection.from_df(df, UserIDKey)
    test.add_from(part)
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
sns.catplot(metrics, x='model', y='value', col='metric', kind='bar')
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
sns.catplot(pred_results.list_metrics().reset_index(), x='model', y='RMSE', kind='bar')
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
