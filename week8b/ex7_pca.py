import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from skimage import img_as_float
import featureNormalize as fn
import pca as pca
import runkMeans as rk
import projectData as pd
import recoverData as rd
import displayData as disp
import kMeansInitCentroids as kmic
import runkMeans as km

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

'''第1部分 加载数据集 并可视化'''
#小数据集方便可视化
print('Visualizing example dataset for PCA.')

data = scio.loadmat('ex7data1.mat')
X = data['X'] #两个特征 

# 可视化
plt.figure()
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b', s=20)
plt.axis('equal')
plt.axis([0.5, 6.5, 2, 8])

input('Program paused. Press ENTER to continue')


'''第2部分 实现PCA 进行数据压缩'''

print('Running PCA on example dataset.')

# 在PCA之前 要对特征进行缩放
X_norm, mu, sigma = fn.feature_normalize(X)

# 执行PCA 返回U矩阵  和S矩阵
U, S = pca.pca(X_norm)

#对比两个不同的特征向量 U[:,0]更好 投影误差最小  U中的各个特征向量（列）都是正交的  2D->1D 取前1个特征向量 作为Ureduce
rk.draw_line(mu, mu + 1.5 * S[0] * U[:, 0]) 
rk.draw_line(mu, mu + 1.5 * S[1] * U[:, 1])

print('Top eigenvector: \nU[:, 0] = {}'.format(U[:, 0])) #利用PCA得到的特征向量矩阵Ureduce（降维后子空间的基）
print('You should expect to see [-0.707107 -0.707107]')

input('Program paused. Press ENTER to continue')

'''第3部分 得到降维后的样本点 再进行压缩重放'''
print('Dimension reductino on example dataset.')

# 可视化特征缩放后的数据集
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], facecolors='none', edgecolors='b', s=20)
plt.axis('equal')
plt.axis([-4, 3, -4, 3])

# 将2维数据映射到1维
K = 1
Z = pd.project_data(X_norm, U, K)
print('Projection of the first example: {}'.format(Z[0]))
print('(this value should be about 1.481274)')

X_rec = rd.recover_data(Z, U, K) #将降维后的1维数据 转换为2维（在特征向量上的投影点）

print('Approximation of the first example: {}'.format(X_rec[0]))
print('(this value should be about [-1.047419 -1.047419])')

# 画出特征缩放后的样本在特征向量上的投影点 并在2者之间连线
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r', s=20)
for i in range(X_norm.shape[0]):
    rk.draw_line(X_norm[i], X_rec[i])

input('Program paused. Press ENTER to continue')

'''第4部分 加载并可视化人脸数据集'''

print('Loading face dataset.')

data = scio.loadmat('ex7faces.mat')
X = data['X'] #得到输入特征矩阵  
print(X.shape[1]) #特征为1024维
disp.display_data(X[0:100]) #可视化前100个人脸

input('Program paused. Press ENTER to continue')


'''第5部分 可视化人脸数据的特征向量'''
print('Running PCA on face dataset.\n(this might take a minute or two ...)')

X_norm, mu, sigma = fn.feature_normalize(X) #对输入特征矩阵进行特征缩放

#执行PCA算法
U, S = pca.pca(X_norm)

#可视化前36个特征向量（每个向量1024维）
disp.display_data(U[:, 0:36].T)

input('Program paused. Press ENTER to continue')

'''第6部分 对人脸数据进行降维 从1024维降到100维'''
print('Dimension reduction for face dataset.')

K = 100
Z = pd.project_data(X_norm, U, K)  #得到降维后的特征矩阵（样本点）

print('The projected data Z has a shape of: {}'.format(Z.shape)) #m*100

input('Program paused. Press ENTER to continue')

'''第7部分 可视化降维后,再压缩重放的人脸数据和原始数据比较'''
print('Visualizing the projected (reduced dimension) faces.')

K = 100
X_rec = rd.recover_data(Z, U, K) #压缩重放

#可视化原始数据
disp.display_data(X_norm[0:100])
plt.title('Original faces')
plt.axis('equal')

#压缩到100维  再压缩重放后的数据
disp.display_data(X_rec[0:100])
plt.title('Recovered faces')
plt.axis('equal')

input('Program paused. Press ENTER to continue')


'''第8部分 利用PCA可视化高维数据'''
image = io.imread('bird_small.png') #读取图片
image = img_as_float(image)

img_shape = image.shape

X = image.reshape((img_shape[0] * img_shape[1], 3))  #将图片格式转换为包含3列（3个颜色通道）的矩阵
K = 16   #聚类中心数量
max_iters = 10  #外循环迭代次数
initial_centroids = kmic.kmeans_init_centroids(X, K)  #初始化K个聚类中心
centroids, idx = km.run_kmeans(X, initial_centroids, max_iters, False) #执行k-means，得到最终的聚类中心和每个样本点所属的聚类中心序号


selected = np.random.randint(X.shape[0], size=1000) #随机选择1000（可以更改）个样本点 每个样本点3维

#可视化3维数据  不同颜色表示每个样本点的所属的簇
cm = plt.cm.get_cmap('RdYlBu')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[selected, 0], X[selected, 1], X[selected, 2], c=idx[selected],cmap=cm, s=15, vmin=0, vmax=K)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

input('Program paused. Press ENTER to continue')

#利用PCA把3维数据 降至2维 进行可视化

X_norm, mu, sigma = fn.feature_normalize(X)  #对特征矩阵X 进行特征缩放

#调用pca 3D->2D
U, S = pca.pca(X_norm)
Z = pd.project_data(X_norm, U, 2)  #得到降维后的特征矩阵

# 可视化2维数据  不同颜色表示每个样本点的所属的簇
plt.figure()
plt.scatter(Z[selected, 0], Z[selected, 1], c=idx[selected].astype(np.float64), cmap=cm,s=15)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')

input('ex7_pca Finished. Press ENTER to exit')
