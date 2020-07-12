import numpy as np


def kmeans_init_centroids(X, K):
    #随机初始化聚类中心
    centroids = np.zeros((K, X.shape[1]))  

    #初始化聚类中心为数据集中的样本点
    centroids=X[np.random.randint(0,X.shape[0],K)]

    return centroids
