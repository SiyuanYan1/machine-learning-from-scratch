import matplotlib.pyplot as plt
import numpy as np
import multivariateGaussian as mvg

#可视化训练集的概率分布  画出等高线图
def visualize_fit(X, mu, sigma2):
    grid = np.arange(0, 35.5, 0.5) #生成网格点
    x1, x2 = np.meshgrid(grid, grid)

    Z = mvg.multivariate_gaussian(np.c_[x1.flatten('F'), x2.flatten('F')], mu, sigma2) #得到每个网格点的概率
    Z = Z.reshape(x1.shape, order='F') 

    plt.figure() #画出训练集样本
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='b', s=15, linewidth=1)

    if np.sum(np.isinf(X)) == 0:  #画出训练集概率分布的等高线图
        lvls = 10 ** np.arange(-20, 0, 3).astype(np.float)
        plt.contour(x1, x2, Z, levels=lvls, colors='r', linewidths=0.7)
