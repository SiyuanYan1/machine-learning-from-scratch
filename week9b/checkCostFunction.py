import numpy as np
import computeNumericalGradient as cng
import cofiCostFunction as ccf

def check_cost_function(lmd):

    #构建一个小数据集 进行梯度检查
    x_t = np.random.rand(4, 3)  #构建电影的特征向量矩阵 4部电影 每部电影3个特征
    theta_t = np.random.rand(5, 3)#构建用户的喜好向量矩阵 5位用户 对应的3个特征

    
    Y = np.dot(x_t, theta_t.T)  # 4x5 得到评分矩阵
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0 #去掉一部分评分值（将一部分随机生成的评分值置0）
    R = np.zeros(Y.shape)  #R矩阵与Y同大小
    R[Y != 0] = 1  #Y矩阵有评分的位置 R[i][j]=1


    x = np.random.randn(x_t.shape[0], x_t.shape[1]) #存放电影特征向量梯度的矩阵
    theta = np.random.randn(theta_t.shape[0], theta_t.shape[1]) #存放用户喜好向量梯度的矩阵
    num_users = Y.shape[1]  #5
    num_movies = Y.shape[0]  #4
    num_features = theta_t.shape[1] #3

    def cost_func(p):
        return ccf.cofi_cost_function(p, Y, R, num_users, num_movies, num_features, lmd)

    #返回每个参数的近似梯度  可以理解为2维空间中参数加减一个很小的数得到的弦的斜率（梯度）
    numgrad = cng.compute_numerial_gradient(cost_func, np.concatenate((x.flatten(), theta.flatten()))) #把两个梯度矩阵转型为一个一维向量 计算每一个分量的近似梯度
     
    #返回每个参数的真实梯度  可以理解为2维空间中参数的切线斜率（梯度）
    cost, grad = ccf.cofi_cost_function(np.concatenate((x.flatten(), theta.flatten())), Y, R, num_users, num_movies, num_features, lmd)
    #对2者进行比较
    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.\n'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient')
    #如果该数值非常小 即2者非常接近 说明没有问题
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If you backpropagation implementation is correct, then\n'
          'the relative difference will be small (less than 1e-9).\n'
          'Relative Difference: {:0.3e}'.format(diff))
