import numpy as np


def feature_normalize(X):
    mu = np.mean(X, 0)  #对特征矩阵每一列求均值
    sigma = np.std(X, 0, ddof=1)  #特征矩阵每一列求标准差
    X_norm = (X - mu) / sigma  #特征矩阵每一列的元素减去该列均值  除以该列标准差  得到特征缩放后的矩阵

    return X_norm, mu, sigma
