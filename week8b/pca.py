import numpy as np
import scipy


def pca(X):
   
    (m, n) = X.shape  #m 样本数  n特征数

    U = np.zeros((n,n)) #U 为n*n的矩阵
    S = np.zeros(n)  #S也是n*n的对角矩阵  只不过svd返回的是其对角线的非0元素 
    #计算协方差矩阵
    Sigma=(1/m)*(X.T.dot(X))
    #对协方差矩阵进行奇异值分解
    U,S,V=scipy.linalg.svd(Sigma)
   
    return U, S
