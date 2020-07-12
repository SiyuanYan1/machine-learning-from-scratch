import numpy as np


def compute_centroids(X, idx, K):
   
    (m, n) = X.shape #m为样本数 n为每个样本的特征数

    centroids = np.zeros((K, n)) #存储新的聚类中心的位置 

    for i in range(K):
        centroids[i]=np.mean(X[idx==i],axis=0)   #对每个簇 计算新的聚类中心 axis=0对每一列求均值

    return centroids
