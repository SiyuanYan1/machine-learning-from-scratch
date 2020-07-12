import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def warmupExercise():
    E5 = np.eye(5)  # eye(5)代表5阶单位阵
    print(E5)
    print('这是一个五阶单位阵')

def plotData(x,y):
    plt.title('Scatter plot of training data')
    plt.scatter(x,y,s=50,cmap='Blues',alpha=0.3)
    plt.xlabel('population')
    plt.ylabel('profit')


def hypothesis(x,theta):
    return x.dot(theta)

def compute_cost(x,y,theta):
    m=y.size
    h=hypothesis(x,theta)#m*1
    cost=(h-y).dot(h-y) /(2*m)
    return cost


# theta=theta-alpha/m * sigma(i=1 to m) (h-y)**2
def gradient_descent_multi(theta,alpha,x,y,num_iters):
    m=len(y)

    j_history=np.zeros(num_iters)# num_iters 次代价函数
    for i in range(num_iters):
        theta=theta-(alpha/m)* (hypothesis(x,theta)-y).dot(x)

        j_history[i]=compute_cost(x,y,theta)

    return theta,j_history








#   2.1 plotting the traning data
print("plotting data")
data=np.loadtxt('ex1data1.txt',delimiter=',',usecols=(0,1))
X=data[:,0]  #feature
y=data[:,1]  #label
m=len(y)     #样本数
#plt.ion()
plt.figure(0)
plotData(X,y)

# 2.2 gradient descent
X=np.c_[np.ones(m),X]
theta=np.zeros(2) #theta0,theta1
iterations=1500
alpha=0.01
print('Initial cost : ' + str(compute_cost(X, y, theta)) + ' (This value should be about 32.07)')
theta,j_history=gradient_descent_multi(theta,alpha,X,y,iterations)

print('Theta found by gradient descent: ' +str(theta))
# 在数据集上绘制出拟合的直线
plt.figure(0)
line1,=plt.plot(X[:,1],np.dot(X,theta),label="Linear Regression")
plt.legend(handles=[line1])
input('Program paused. Press ENTER to continue')
predict1=np.dot(np.array([1,3.5]),theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))
#'''第3部分 可视化代价函数'''
print('Visualizing J(theta0, theta1) ...')

theta0_vals = np.linspace(-10, 10, 100)  #参数1的取值
theta1_vals = np.linspace(-1, 4, 100)    #参数2的取值

xs, ys = np.meshgrid(theta0_vals, theta1_vals)  # 生成网格
J_vals = np.zeros(xs.shape)

for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)  # 计算每个网格点的代价函数值

J_vals = np.transpose(J_vals)

fig1 = plt.figure(1)  # 绘制3d图形
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

# 绘制等高线图 相当于3d图形的投影
plt.figure(2)
lvls = np.logspace(-2, 3, 20)
plt.contour(xs, ys, J_vals, levels=lvls)
plt.show()










