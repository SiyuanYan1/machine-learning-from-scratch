# 系统自带包
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio  # 读取.mat文件

# 手写的包
import displayData as dd  # 可视化数据
import predict as pd

input_layer_size = 400  # 输入层的单元数  原始输入特征数 20*20=400
hidden_layer_size = 25  # 隐藏层 25个神经元
num_labels = 10  # 10个标签 数字0对应类别10  数字1-9对应类别1-9

'''第1部分 加载数据集并可视化'''

print('Loading and Visualizing Data ...')

data = scio.loadmat('ex3data1.mat')  # 读取数据
X = data['X']  # 获取输入特征矩阵 5000*400
y = data['y'].flatten()  # 获取5000个样本的标签 用flatten()函数 将5000*1的2维数组 转换成包含5000个元素的一维数组
m = y.size  # 样本数 5000

# 随机选100个样本 可视化
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

dd.display_data(selected)

'''第2部分 加载训练好的神经网络参数'''

print('Loading Saved Neural Network Parameters ...')

data = scio.loadmat('ex3weights.mat')  # 读取参数数据
# 本实验神经网络结构只有3层 输入层，隐藏层 输出层
theta1 = data['Theta1']  # 输入层和隐藏层之间的参数矩阵
print(theta1.size)
theta2 = data['Theta2']  # 隐藏层和输出层之间的参数矩阵
print(theta2.size)
'''第3部分 利input1=tf.placeholder(tf.float32)用训练好的参数 完成神经网络的前向传播 实现预测过程'''
pred = pd.predict(theta1, theta2, X)

print('Training set accuracy: {}'.format(np.mean(pred == y) * 100))

