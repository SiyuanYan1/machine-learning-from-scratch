import numpy as np


def project_data(X, U, K): #得到降维后的样本点

    Z = np.zeros((X.shape[0], K)) #降维后的特征矩阵 Z：m*K X：m*n

    Z=X.dot(U[:,:K])

    return Z
