"""
Basic algorithm definitions as starting points.
"""

from lenskit.algorithms import item_knn, user_knn, als, funksvd

II = item_knn.ItemItem(20, save_nbrs=2500)
UU = user_knn.UserUser(30)
MF = als.BiasedMF(50)
WRMF = als.ImplicitMF(50)
MFSGD = funksvd.FunkSVD(50)