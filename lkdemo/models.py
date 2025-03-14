"""
This module defines the various models, and their default configurations, that
we are going to test in our experiment.
"""

from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from lenskit.basic import BiasScorer, PopScorer
from lenskit.knn import ItemKNNScorer, UserKNNScorer

Bias = BiasScorer(damping=5)
Pop = PopScorer()
IIE = ItemKNNScorer(max_nbrs=20, save_nbrs=2500, feedback="explicit")
UUE = UserKNNScorer(max_nbrs=30)
ALS = BiasedMFScorer(embedding_size=50)
IALS = ImplicitMFScorer(embedding_size=50)
III = ItemKNNScorer(max_nbrs=20, save_nbrs=2500, feedback="implicit")
