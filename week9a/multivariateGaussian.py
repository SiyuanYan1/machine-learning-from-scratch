import numpy as np


def multivariate_gaussian(X, mu, sigma2):
    k = mu.size  #特征数
    #如果sigma2 转换成对角矩阵（协方差矩阵）
    #当协方差矩阵是对角矩阵时  单元高斯分布乘积和多元高斯分布是等价的
    if sigma2.ndim == 1 or (sigma2.ndim == 2 and (sigma2.shape[1] == 1 or sigma2.shape[0] == 1)):
        sigma2 = np.diag(sigma2)

    x = X - mu #对原始特征矩阵进行均值规范化
    #此时单元高斯分布乘积和多元高斯分布是等价的  所以直接用多元高斯分布公式得到训练集的概率分布
    p = (2 * np.pi) ** (-k / 2) * np.linalg.det(sigma2) ** (-0.5) * np.exp(-0.5*np.sum(np.dot(x, np.linalg.pinv(sigma2)) * x, axis=1))

    return p
