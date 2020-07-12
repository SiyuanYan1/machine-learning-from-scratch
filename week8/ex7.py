import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import misc
from skimage import io
from skimage import img_as_float

import runkMeans as km
import findClosestCentroids as fc
import computeCentroids as cc
import kMeansInitCentroids as kmic

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

'''第1部分 为每个样本点找到离他最近的聚类中心'''

print('Finding closest centroids.')

data = scio.loadmat('ex7data2.mat') #加载矩阵格式的数据
X = data['X']  #提取输入特征矩阵

print(len(X))
k = 3  # 随机初始化3个聚类中心
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

#找到离每个样本最近的初始聚类中心序号
idx = fc.find_closest_centroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: ')
print('{}'.format(idx[0:3]))
print('(the closest centroids should be 0, 2, 1 respectively)')

input('Program paused. Press ENTER to continue')

'''第2部分 更新聚类中心'''

print('Computing centroids means.')

centroids = cc.compute_centroids(X, idx, k) #在簇分配结束后 对每个簇的样本点重新计算聚类中心

print('Centroids computed after initial finding of closest centroids: \n{}'.format(centroids))
print('the centroids should be')
print('[[ 2.428301 3.157924 ]')
print(' [ 5.813503 2.633656 ]')
print(' [ 7.119387 3.616684 ]]')

input('Program paused. Press ENTER to continue')

'''第3部分 运行k-means聚类算法'''
print('Running K-Means Clustering on example dataset.')

#加载数据集
data = scio.loadmat('ex7data2.mat') 
X = data['X']


K = 3   #聚类中心数量
max_iters = 10  #设置外循环迭代次数

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])  #初始化聚类中心

centroids, idx = km.run_kmeans(X, initial_centroids, max_iters, True) #运行k-means算法 返回最终聚类中心位置即每个样本点所属的聚类中心
#并把中间过程以及最终聚类效果可视化
print('K-Means Done.')

input('Program paused. Press ENTER to continue')

'''第4部分 运行k-means聚类算法 压缩图片'''
print('Running K-Means clustering on pixels from an image')

#加载图片
image = io.imread('bird_small.png')
image = img_as_float(image)

# 图片大小
img_shape = image.shape


X = image.reshape(img_shape[0] * img_shape[1], 3) #把图片转换成3个列向量构成的矩阵  每个列向量代表每个颜色通道的所有像素点 

#可以设置不同的参数 观察效果
K = 16 #聚类中心数量
max_iters = 10 #外循环迭代次数

#初始化聚类中心位置很重要  初始化不同  最终聚类效果也会不同
initial_centroids = kmic.kmeans_init_centroids(X, K) 

# 运行k-means
centroids, idx = km.run_kmeans(X, initial_centroids, max_iters, False) #False不进行可视化

print('K-Means Done.')

input('Program paused. Press ENTER to continue')


print('Applying K-Means to compress an image.')

# 得到最终聚类结束后 每个样本所属的聚类中心序号
idx = fc.find_closest_centroids(X, centroids)

#用idx做索引
idx=idx.astype(int) #将数值类型转换为整型
idx=idx.tolist()  #将数组转换为列表  

X_recovered = centroids[idx]  #将每个样本点位置转换为它所属簇的聚类中心的位置  实现压缩

X_recovered = np.reshape(X_recovered, (img_shape[0], img_shape[1], 3)) #把图像转换为之前的维度
#misc.imsave('compress.png',X_recovered)
io.imsave('compress.png',X_recovered)
plt.subplot(2, 1, 1)  #可视化原始图片
plt.imshow(image)
plt.title('Original')

plt.subplot(2, 1, 2)  #压缩后的图片
plt.imshow(X_recovered)
plt.title('Compressed, with {} colors'.format(K))

input('ex7 Finished. Press ENTER to exit')
