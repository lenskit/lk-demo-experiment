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


# Setup


Below are the list of packages required to successfully run the analysis. They are divided into partitions to signify their specific task.<br>
We need the pathlib package for working with files and folders

```python
from pathlib import Path
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

Logging to show what's happening in LensKit routines:

```python
from lenskit import util
util.log_to_notebook()
```

```python
import logging
_log = logging.getLogger('eval-report')
```

We will use lenskit for training, running, and evaluating recommender algorithms

```python
from lenskit import topn
from lenskit.metrics.predict import rmse
```

# Load Data


We specify the dataset we will use for our analysis and the main directory from where we read the recommendation and prediction files. From the main directory we find all the directories associated with the dataset and then read the recommendation and predictions files from those directories.

```python tags=["parameters"]
dataset = "ml100k"
```

```python
output_root = Path("runs")
```

```python
dirs = [fld for fld in output_root.glob(f'{dataset}-*')]
```

```python
recs = []
for fld in dirs:
    for file in fld.glob("recs-*"):
        rec = pd.read_parquet(file)
        rec["algorithm"] = fld.name.split("-")[1]
        recs.append(rec)

recs = pd.concat(recs, ignore_index=True)
recs = recs.astype({'algorithm': 'category'})
recs.info()
```

```python
rec_algos = recs['algorithm'].unique()
rec_algos
```

```python
preds = []
for fld in dirs:
    for file in fld.glob("pred-*"):
        pred = pd.read_parquet(file)
        pred["algorithm"] = fld.name.split("-")[1]
        preds.append(pred)

preds = pd.concat(preds, ignore_index=True)
preds = preds.astype({'algorithm': 'category'})
preds.info()
```

We need to load the test data so that we have the ground truths for computing accuracy

```python
split_root = Path("data-split")
split_dir = split_root / dataset
```

```python
test = []
for file in split_dir.glob("test-*.parquet"):
    test.append(pd.read_parquet(file,).assign(part=file.stem.replace('.parquet', '')))

test = pd.concat(test).rename(columns={
    'user_id': 'user',
    'item_id': 'item',
})
test.head()
```

# Top-N Metrics


The topn.RecListAnalysis class computes top-N metrics for recommendation list and takes care of making sure that the recommendations and ground truths are properly matched. Refer to the documentation for detailed explanation of the purpose for the RecListAnalysis class and how the analysis is done - https://lkpy.lenskit.org/en/stable/evaluation/topn-metrics.html

```python
rla = topn.RecListAnalysis()

rla.add_metric(topn.precision)
rla.add_metric(topn.recip_rank)
rla.add_metric(topn.ndcg)
results = rla.compute(recs, test.drop(columns=['rating']), include_missing=True)
results = results.fillna(0)
results.head()
```

We will reshape the 'results' dataframe by stacking the columns to index and then use the bar chart to visualize the performance of our algorithms with respect to the precision, reciprocal rank and ndcg metrics

```python
pltData = (results.drop(columns=['nrecs', 'ntruth']).stack()).reset_index()
pltData.columns = ['algorithm', 'user', 'metric', 'val']
pltData.head()
```

We need to determine if the differences we observe in the performances of the algorithms for the various metrics are statistically significant. To achieve this, we will need to use either a parametric or non-parametric statistical test for comparing the differences. We will consider a parametric test - repeated ANOVA measure cause our sample groups are correlated.

```python
g = sns.catplot(x = "algorithm", y = "val", data = pltData, kind="bar", col = "metric", aspect=1.2, height=3, sharey=False)
```

## Prediction RMSE

We will also look at the prediction RMSE.

```python
preds = preds.rename(columns={'score': 'prediction'})
user_rmse = preds.groupby(['algorithm', 'user']).apply(lambda df: rmse(df['prediction'], df['rating']))
user_rmse = user_rmse.reset_index(name='RMSE')
```

```python
sns.catplot(x='algorithm', y='RMSE', data=user_rmse, kind='bar')
```
