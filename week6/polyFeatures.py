import numpy as np
def poly_features(X,p):
    X_poly=X[:] #第一列为原始特征

    for i in range(2,p+1):
        X_poly=np.c_[X_poly,X**i]
    return X_poly