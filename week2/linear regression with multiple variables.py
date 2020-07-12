import numpy as np
import matplotlib.pyplot as plt
def feature_normalize(X):
    n=X.shape[1]
    mu=np.mean(X,axis=0)  #压缩行，求每一列的值 1*2
    sigma=np.std(X,axis=0)  #1*2
    X_norm=(X-mu)/sigma
    return X_norm,mu,sigma
def h(x,theta):
    return x.dot(theta)
def compute_cost(x,y,theta):
    m=y.size
    cost=(h(x,theta)-y).dot(h(x,theta)-y)/(2*m)
    return cost

def gradient_descent_multi(x,y,theta,alpha,iterations):
    m=y.size
    j_history=np.zeros(iterations)
    for i in range(iterations):
        theta=theta-(alpha/m)*(h(x,theta)-y).dot(x)
        j_history[i]=compute_cost(x,y,theta)
    return theta,j_history



# Part 1  Featur scaling
print('Loading Data....')
data=np.loadtxt('ex1data2.txt',delimiter=',',dtype=np.int64)
X=data[:,0:2]
y=data[:,2]
m=y.size

#打印前十个traning samples
print("First 10 examples from the dataset: ")
for i in range(10):
    print("x={},y={}".format(X[i],y[i]))
#input("Program paused. Press ENTER to continue")
print("Normalizing Features")
X,mu,sigma=feature_normalize(X)
# Part 2 Gradient Descent

print('Running gradient descent ...')
X=np.c_[np.ones(m),X]
alpha = 0.03  # 学习率
num_iters = 400  # 迭代次数
theta=np.zeros(3)
theta,j_history=gradient_descent_multi(X,y,theta,alpha,num_iters)
# 绘制代价函数值随迭代次数的变化曲线
plt.figure()
plt.plot(np.arange(j_history.size),j_history)
plt.xlabel("number of iterations")
plt.ylabel("J cost")

print("the theta should be {}".format(theta))
x1=np.array([1650,3])
x1=(x1-mu)/sigma  #***预测时的x也要特征缩放***
x1=np.r_[np.array([1]),x1]
y_predict=h(x1,theta)
print("The profit should be {}".format(y_predict))
plt.show()





