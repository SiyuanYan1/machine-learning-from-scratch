import numpy as np


def find_closest_centroids(X, centroids):
    
    K = centroids.shape[0]  #聚类中心数量

    m = X.shape[0]  #样本数

  
    idx = np.zeros(m) #存储m个样本对应的最近的聚类中心序号

    for i in range(m):
        a=(X[i]-centroids).dot((X[i]-centroids).T)  #得到一个方阵  对角线上的元素为该样本点到每个聚类中心的距离
        idx[i]=np.argsort(a.diagonal())[0]  #取出对角线元素 对其索引进行排序  返回离该样本最近的聚类中心的序号

    return idx
