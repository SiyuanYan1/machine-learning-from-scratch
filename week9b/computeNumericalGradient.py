import numpy as np


def compute_numerial_gradient(cost_func, theta):
    #theta存放所有参数 包括所有特征向量和喜好向量的每一个分量
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)

    e = 1e-4
    #用弦的斜率（梯度）与切线的斜率（梯度）进行比较  来进行梯度检查 如果差不多 就说明没问题
    #对每个参数都进行梯度检查 
    for p in range(theta.size): 
        perturb[p] = e
        #返回弦两个端点的代价
        loss1, grad1 = cost_func(theta - perturb) 
        loss2, grad2 = cost_func(theta + perturb)

        numgrad[p] = (loss2 - loss1) / (2 * e) #弦的斜率（梯度）
        perturb[p] = 0

    return numgrad
