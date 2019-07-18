from lenskit.algorithms import als

E_LU = als.BiasedMF(50, method='lu')
E_CD = als.BiasedMF(50, method='cd')

I_LU = als.ImplicitMF(50, method='lu')
I_CG = als.ImplicitMF(50, method='cg')
