import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import findClosestCentroids as fc
import computeCentroids as cc


def run_kmeans(X, initial_centroids, max_iters, plot): #plot设置是否进行可视化 
    if plot:
        plt.figure()

    (m, n) = X.shape #m样本数  n样本特征数
    K = initial_centroids.shape[0]  #聚类中心数量
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)  #存放每个样本所属的聚类中心序号

    # 运行k-means
    for i in range(max_iters):  #外循环
        print('K-Means iteration {}/{}'.format((i + 1), max_iters))  

        idx = fc.find_closest_centroids(X, centroids) #第一个内循环 为每个样本找到最近的聚类中心
        
        if plot:
            plot_progress(X, centroids, previous_centroids, idx, K, i) #画出此时簇分配的状态
            previous_centroids = centroids
            input('Press ENTER to continue')

        centroids = cc.compute_centroids(X, idx, K) #第2个内循环  更新聚类中心

    return centroids, idx  #返回最终聚类中心的位置  和每个样本所属的聚类中心序号


def plot_progress(X, centroids, previous, idx, K, i):
    plt.scatter(X[:, 0], X[:, 1], c=idx, s=15)   #不同聚类中心用不同的颜色表示 

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', s=25) #标出聚类中心

    for j in range(centroids.shape[0]):  #为更新后的聚类中心和之前的聚类中心连线
        draw_line(centroids[j], previous[j])

    plt.title('Iteration number {}'.format(i + 1))


def draw_line(p1, p2):
    plt.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), c='black', linewidth=1)
