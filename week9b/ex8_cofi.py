import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.optimize as opt

import cofiCostFunction as ccf
import checkCostFunction as cf
import loadMovieList as lm
import normalizeRatings as nr


plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

'''第1部分 加载电影评分数据集'''
print('Loading movie ratings dataset.')


data = scio.loadmat('ex8_movies.mat')
Y = data['Y']  #1682*943  1682部电影 943个用户 评分（1-5）   评分矩阵
R = data['R']  #1682*943  1682部电影 943个用户 当且仅当R(i,j)=1时 Y(i,j)有评分



# 计算第一部电影的平均评分
print('Average ratings for movie 0(Toy Story): {:0.6f}/5'.format(np.mean(Y[0, np.where(R[0] == 1)])))

# 可视化评分矩阵
plt.figure()
plt.imshow(Y)
plt.colorbar()
plt.xlabel('Users')
plt.ylabel('Movies')

input('Program paused. Press ENTER to continue')


'''第2部分 计算协同过滤的代价'''

data = scio.loadmat('ex8_movieParams.mat') #加载预训练好的协同过滤的参数
X = data['X']    #电影特征向量构成的矩阵
theta = data['Theta']  #用户喜好向量构成的矩阵
num_users = data['num_users']  #用户数量
num_movies = data['num_movies']  #电影数量
num_features = data['num_features']  #特征数量

#缩减数据集 使运行速度更快
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features] #选择前5部电影的特征向量 其中每个特征向量只取前3个特征
theta = theta[0:num_users, 0:num_features]#选择前4位用户的喜好向量 其中每个向量只取前3个特征
Y = Y[0:num_movies, 0:num_users] #缩减后的评分矩阵
R = R[0:num_movies, 0:num_users] #缩减后的R矩阵

# 根据训练好的参数向量  计算此时的代价(不带正则化 lmd=0)
cost, grad = ccf.cofi_cost_function(np.concatenate((X.flatten(), theta.flatten())), Y, R, num_users, num_movies, num_features, 0)

#验证程序正确性
print('Cost at loaded parameters: {:0.2f}\n(this value should be about 22.22)'.format(cost))

input('Program paused. Press ENTER to continue')


'''第3部分 进行梯度检查'''

print('Checking gradients (without regularization) ...')

#构建小数据集  对每一个参数进行梯度检查 （可以理解为 弦的斜率是否近似于切线的斜率）
cf.check_cost_function(0)

input('Program paused. Press ENTER to continue')

'''第4部分 计算协同过滤的代价 带正则化'''
#lmd=1.5
cost, _ = ccf.cofi_cost_function(np.concatenate((X.flatten(), theta.flatten())), Y, R, num_users, num_movies, num_features, 1.5)

#验证程序正确性
print('Cost at loaded parameters (lambda = 1.5): {:0.2f}\n'
      '(this value should be about 31.34)'.format(cost))

input('Program paused. Press ENTER to continue')

'''第5部分 进行梯度检查 （带正则化）'''
#计算带正则化的近似梯度和梯度 并进行比较
print('Checking Gradients (with regularization) ...')

#lmd=1.5
cf.check_cost_function(1.5)

input('Program paused. Press ENTER to continue')

'''第6部分 增加一个新用户 并对所有的1682部电影进行评分'''
movie_list = lm.load_movie_list() #读取电影名放在列表里

# 初始新用户对每部电影的评分为0 表示新用户都没看过
my_ratings = np.zeros(len(movie_list))

#新用户对其中看过的部分电影进行评分（1-5）
#比如新用户看过第一部电影 并给他的评分为4 其他类似
my_ratings[0] = 4

my_ratings[97] = 2

my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('New user ratings:\n')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

input('Program paused. Press ENTER to continue')

'''第7部分 在原始电影评分数据集上，增加新用户的评分 即增加一列数据 重新训练协同过滤算法'''
print('Training collaborative filtering ...\n'
      '(this may take 1 ~ 2 minutes)')


# 加载原始数据
data = scio.loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
# 943 users
#
# R is a 1682x943 matrix, where R[i,j] = 1 if and only if user j gave a
# rating to movie i

# 加上新用户的数据 构建新的电影评分数据集
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0), R]

# 规范化评分矩阵 计算每部电影的平均评分 原始评分矩阵每一行减去各自的平均评分 返回新的评分矩阵
Ynorm, Ymean = nr.normalize_ratings(Y, R)

num_users = Y.shape[1] #用户数
num_movies = Y.shape[0] #电影数
num_features = 10  #特征向量取10个特征

# 初始化所有参数为很小的随机值
X = np.random.randn(num_movies, num_features)
theta = np.random.randn(num_users, num_features)

initial_params = np.concatenate([X.flatten(), theta.flatten()]) #把所有参数放在一个一维向量中

lmd = 10 #正则化系数


def cost_func(p):  #返回协同过滤的代价
    return ccf.cofi_cost_function(p, Ynorm, R, num_users, num_movies, num_features, lmd)[0]


def grad_func(p):  #返回协同过滤的梯度
    return ccf.cofi_cost_function(p, Ynorm, R, num_users, num_movies, num_features, lmd)[1]

#调用高级优化算法 进行训练 得到最优的参数
theta, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=initial_params, maxiter=100, disp=False, full_output=True)

# 把训练好的参数 再重新转型为矩阵形式
X = theta[0:num_movies * num_features].reshape((num_movies, num_features))
theta = theta[num_movies * num_features:].reshape((num_users, num_features))

#打印用户的喜好向量矩阵
print('Recommender system learning completed')
print(theta)

input('Program paused. Press ENTER to continue')


'''第8部分 根据训练好的参数 预测评分矩阵的缺失值'''

p = np.dot(X, theta.T) #X训练好的每部电影的特征向量构成的矩阵 theta训练好的每位用户的喜好向量构成的矩阵  得到预测的评分矩阵
my_predictions = p[:, 0] + Ymean #得到预测的第一位新添加用户对每部电影的评价 别忘了加上减去的每部电影的平均评分

indices = np.argsort(my_predictions)[::-1] #得到预测的新用户对每部电影的评价的排序 从大到小  返回电影的索引
print('\nTop recommendations for you:')  #为新用户推荐的前10位的电影
for i in range(10):
    j = indices[i]
    print('Predicting rating {:0.1f} for movie {}'.format(my_predictions[j], movie_list[j])) #打印预测评分+电影名称

print('\nOriginal ratings provided:') #新用户对部分电影的原始评分+对应的电影名
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

input('ex8_cofi Finished. Press ENTER to exit')
