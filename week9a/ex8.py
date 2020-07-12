import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import estimateGaussian as eg
import multivariateGaussian as mvg
import visualizeFit as vf
import selectThreshold as st

plt.ion()
# np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

'''第1部分 加载示例数据集'''

#先通过一个小数据集进行异常检测  便于可视化

# 数据集包含两个特征 
# 一些机器的等待时间和吞吐量  实验目的找出其中可能有异常的机器


print('Visualizing example dataset for outlier detection.')


data = scio.loadmat('ex8data1.mat')
X = data['X']#训练集样本特征矩阵
Xval = data['Xval']  #验证集样本特征矩阵
yval = data['yval'].flatten() #验证集样本标签 异常/正常 

# 可视化样例训练集
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='b', marker='x', s=15, linewidth=1)
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')  #x1等待时间
plt.ylabel('Throughput (mb/s') #x2吞吐量
# plt.ioff()
# plt.show()


input('Program paused. Press ENTER to continue')

'''第2部分 估计训练集的分布'''
# 假设数据集的各个特
# 征服从高斯分布

print('Visualizing Gaussian fit.')

# 参数估计  计算每个特征的均值和方差
mu, sigma2 = eg.estimate_gaussian(X)

# 基于单元高斯分布得到训练集的概率分布
p = mvg.multivariate_gaussian(X, mu, sigma2)
#可视化训练集的概率分布  画出等高线图
vf.visualize_fit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s')
input('Program paused. Press ENTER to continue')

'''第3部分 基于验证集 得到一个最好的概率分布阈值'''
pval = mvg.multivariate_gaussian(Xval, mu, sigma2) #根据训练集的概率分布 得到验证集样本的概率

epsilon, f1 = st.select_threshold(yval, pval)   #选择合适的概率阈值
print('Best epsilon found using cross-validation: {:0.4e}'.format(epsilon))
print('Best F1 on Cross Validation Set: {:0.6f}'.format(f1))
print('(you should see a value epsilon of about 8.99e-05 and F1 of about 0.875)')

# 标出训练集中的异常值
outliers = np.where(p < epsilon)
plt.scatter(X[outliers, 0], X[outliers, 1], marker='o', facecolors='none', edgecolors='r')

input('Program paused. Press ENTER to continue')


'''第4部分 基于大数据集 进行异常检测（特征数很多）'''
data = scio.loadmat('ex8data2.mat')
X = data['X'] #训练集样本特征矩阵
Xval = data['Xval'] #验证集样本特征矩阵
yval = data['yval'].flatten() #验证集样本标签 1异常 0正常

#使用基于单元高斯分布的异常检测模型 
# 计算每一个特征的均值和方差
mu, sigma2 = eg.estimate_gaussian(X)

# 将特征方差转换为对角矩阵作为协方差矩阵 代入多元高斯分布公式  得到训练集的概率分布
p = mvg.multivariate_gaussian(X, mu, sigma2)

# 得到验证集每个样本的概率
pval = mvg.multivariate_gaussian(Xval, mu, sigma2)

# 选择一个最好的阈值
epsilon, f1 = st.select_threshold(yval, pval)

#验证程序正确性
print('Best epsilon found using cross-validation: {:0.4e}'.format(epsilon))
print('Best F1 on Cross Validation Set: {:0.6f}'.format(f1))
print('# Outliers found: {}'.format(np.sum(np.less(p, epsilon))))
print('(you should see a value epsilon of about 1.38e-18, F1 of about 0.615, and 117 outliers)')

input('ex8 Finished. Press ENTER to exit')
