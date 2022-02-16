"""
This module defines the algorithms, and their default configurations, that
we are going to use.
"""

import warnings
from lenskit.algorithms import item_knn, user_knn, als
from lenskit.algorithms import basic

Bias = basic.Bias(damping=5)
Pop = basic.Popular()
IIE = item_knn.ItemItem(20, save_nbrs=2500)
UUE = user_knn.UserUser(30)
ALS = als.BiasedMF(50)
IALS = als.ImplicitMF(50, use_ratings=False)
III = item_knn.ItemItem(20, save_nbrs=2500, aggregate='sum', center=False, use_ratings=False)
