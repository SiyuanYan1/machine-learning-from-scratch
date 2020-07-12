#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:35:12 2018

@author: sduhao
"""
'''系统自带的库'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio   #引入读取.mat文件的库

'''自己编写的库'''
import displayData as dd  #引入可视化数据集的程序
import lrCostFunction as lCF #引入逻辑回归多分类代价函数
import oneVsAll as ova    #引入逻辑回归训练程序
import predictOneVsAll as pova  #引入逻辑回归预测程序

plt.ion()


input_layer_size = 400  # 输入特征的维度 每张数字图片20*20=400
num_labels = 10         # 10个标签 注意“0”对应第十个标签   1-9一次对应第1-9个标签
                       

'''第1部分  加载手写数字训练数据集 并可视化部分训练样本'''

# 加载训练集
print('加载并可视化数据 ...')

data = scio.loadmat('ex3data1.mat')  #读取训练集 包括两部分 输入特征和标签
print(data.keys())
print(data['X'].shape)
print(data['y'].shape)
X = data['X']   #提取输入特征 5000*400的矩阵  5000个训练样本 每个样本特征维度为400 一行代表一个训练样本
# print(X)
y = data['y'].flatten() #提取标签 data['y']是一个5000*1的2维数组 利用flatten()将其转换为有5000个元素的一维数组
                        #y中 数字0对应的标签是10
print(y)

m = y.size  #训练样本的数量

# 随机抽取100个训练样本 进行可视化
rand_indices = np.random.permutation(range(m)) #获取0-4999 5000个无序随机索引
selected = X[rand_indices[0:100], :]  #获取前100个随机索引对应的整条数据的输入特征

dd.display_data(selected)   #调用可视化函数 进行可视化


input('Program paused. Press ENTER to continue')

'''第2-1部分  编写逻辑回归的代价函数(正则化),做简单的测试'''

# 逻辑回归（正则化）代价函数的测试用例
print('Testing lrCostFunction()')

theta_t = np.array([-2, -1, 1, 2]) #初始化假设函数的参数  假设有4个参数
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape((3, 5)).T/10] #输入特征矩阵 5个训练样本 每个样本3个输入特征，前面添加一列特征=1
y_t = np.array([1, 0, 1, 0, 1]) #标签 做2分类

lmda_t = 3  #正则化惩罚性系数
cost, grad = lCF.lr_cost_function(theta_t, X_t, y_t, lmda_t) #传入代价函数 

#返回当前的代价函数值和梯度值  与期望值比较 验证程序的正确性

np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
print('Cost: {:0.7f}'.format(cost))
print('Expected cost: 2.534819')
print('Gradients:\n{}'.format(grad))
print('Expected gradients:\n[ 0.146561 -0.548558 0.724722 1.398003]')

input('Program paused. Press ENTER to continue')


'''第2-2部分  训练多分类的逻辑回归 实现手写数字识别'''
print('Training One-vs-All Logistic Regression ...')

lmd = 0.1 #正则化惩罚项系数
all_theta = ova.one_vs_all(X, y, num_labels, lmd)  #返回训练好的参数

input('Program paused. Press ENTER to continue')


'''第3部分  在训练集上测试 之前训练的逻辑回归多分类器的准确率'''

pred = pova.predict_one_vs_all(all_theta, X) #分类器的预测类别

print('Training set accuracy: {}'.format(np.mean(pred == y)*100))  #与真实类别进行比较 得到准确率

input('ex3 Finished. Press ENTER to exit')
