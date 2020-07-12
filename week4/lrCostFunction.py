import numpy as np
from sigmoid import *


#逻辑回归的假设函数
def h(X,theta):
    return sigmoid(np.dot(X,theta)) #X: m*(n+1)  theta:(n+1,) 内积返回结果(m*1,)

#计算代价函数
def Compute_cost(theta, X, y, lmd):
    m = y.size  #样本数

    cost = 0
    myh=h(X,theta)  #得到假设函数值
    
    term1=-y.dot(np.log(myh)) #y:(m*1,)   log(myh):(m*1,)  得到一个数值
    term2=(1-y).dot(np.log(1-myh))#1-y:(m*1,)   log(1-myh):(m*1,) 得到一个数值
    term3=theta[1:].dot(theta[1:])*lmd #正则化项 注意不惩罚theta0 得到一个数值 thrta[1:] (n,)
    cost=(1/m)*(term1-term2)+(1/(2*m))*term3
   
    return cost

#计算梯度值
def Compute_grad(theta,X,y,lmd):
    m = y.size  #样本数

    grad = np.zeros(theta.shape) #梯度是与参数同维的向量
    myh=h(X,theta)  #得到假设函数值

    reg=(lmd/m)*theta[1:] #reg (n,)
    beta=myh-y    #beta: (m,)
    grad=beta.dot(X)/m #beta:(m,)  X:m*(n+1)  grad:(n+1,)
 
    grad[1:]+=reg
    return grad

def lr_cost_function(theta, X, y, lmd):
   
    cost=Compute_cost(theta, X, y, lmd)
    grad=Compute_grad(theta, X, y, lmd)
    return cost, grad
