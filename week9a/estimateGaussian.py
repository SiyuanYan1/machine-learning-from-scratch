import numpy as np


def estimate_gaussian(X):
    m,n=X.shape#307,2
    mu=np.mean(X,axis=0)
    sigma2=np.var(X,axis=0)

    return mu,sigma2
