"""
This module defines the various models, and their default configurations, that
we are going to test in our experiment.
"""

from lenskit.als import BiasedMF, ImplicitMF
from lenskit.basic import BiasScorer, PopScorer
from lenskit.knn import ItemKNNScorer, UserKNNScorer

Bias = BiasScorer(damping=5)
Pop = PopScorer()
IIE = ItemKNNScorer(20, save_nbrs=2500, feedback="explicit")
UUE = UserKNNScorer(30)
ALS = BiasedMF(50)
IALS = ImplicitMF(50)
III = ItemKNNScorer(20, save_nbrs=2500, feedback="implicit")
