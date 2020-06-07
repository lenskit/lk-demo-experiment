"""
This module defines the algorithms, and their default configurations, that
we are going to use.
"""

from lenskit.algorithms import item_knn, user_knn, als, tf
from lenskit.algorithms import basic

Bias = basic.Bias(damping=5)
Pop = basic.Popular()
II = item_knn.ItemItem(20, save_nbrs=2500)
UU = user_knn.UserUser(30)
ALS = als.BiasedMF(50)
IALS = als.ImplicitMF(50)
BPR = tf.BPR(50)
TFMF = tf.IntegratedBiasMF(50)
