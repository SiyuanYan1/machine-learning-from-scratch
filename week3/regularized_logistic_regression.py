import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
def plot_decision_boundary(theta, X, y):
    plot_data(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need two points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        plt.plot(plot_x, plot_y)

        plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'], loc=1)
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))

        # Evaluate z = theta*x over the grid
        for i in range(0, u.size):
            for j in range(0, v.size):
                z[i, j] = np.dot(map_feature(u[i], v[j]), theta)

        z = z.T

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        cs = plt.contour(u, v, z, levels=[0], colors='r', label='Decision Boundary')
        plt.legend([cs.collections[0]], ['Decision Boundary'])

def plot_data(x,y):
    plt.figure(0)
    positive=x[y==1]
    negative=x[y==0]
    plt.scatter(positive[:,0],positive[:,1],marker='+',c='red',label='y=1')
    plt.scatter(negative[:,0],negative[:,1],marker='o',c='blue',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y=1','y=0'],loc='best')





def map_feature(x1, x2):
    degree = 6
    x1 = x1.reshape((x1.size, 1))
    x2 = x2.reshape((x2.size, 1))
    result = np.ones(x1[:, 0].shape)

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            result = np.c_[result, (x1 ** (i - j)) * (x2 ** j)]  #不断拼接新的列 扩充特征矩阵

    return result

def sigmoid(z):
    return 1/(1+np.exp(-z))

def h(theta,x):
    return sigmoid(x.dot(theta))

def cost_function_reg(theta,x,y,lmd):
    m=y.size
    term1=(-y).dot(np.log(h(theta,x)))
    term2=(1-y).dot(np.log(1-h(theta,x)))
    term3=(lmd/(2*m))*( theta[1:].dot(theta[1:]))  #不惩罚theta0
    cost=(term1-term2)/m+term3
    grad_j=(h(theta,x)-y).dot(x)/m
    grad_j[1:]+=(lmd/m)*theta[1:]
    return cost,grad_j

def predict(theta,x):
    p=h(theta,x)
    p[p>=0.5]=1
    p[p<0.5]=0
    return p



data = np.loadtxt('ex2data2.txt', delimiter=',')  # 加载txt格式训练数据集 每一行用','分隔
X = data[:, 0:2]  # 前两列是原始输入特征（2）
y = data[:, 2]  # 最后一列是标签 0/1

plot_data(X, y)  # 可视化训练集
#plt.show()


input('Program paused. Press ENTER to continue')

'''第1部分 增加新的多项式特征，计算逻辑回归(正则化)代价函数和梯度'''
X = map_feature(X[:, 0], X[:, 1])

initial_theta = np.zeros(X.shape[1])

lmd = 1  # 正则化惩罚项系数

# 计算参数为0时的代价函数值和梯度
cost, grad = cost_function_reg(initial_theta, X, y, lmd)

# 与期望值比较 验证正确性
np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})
print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

input('Program paused. Press ENTER to continue')

test_theta = np.ones(X.shape[1])
# 计算参数非0（1）时的代价函数值和梯度
cost, grad = cost_function_reg(test_theta, X, y, lmd)
# 与期望值比较 验证正确性
print('Cost at test theta: {}'.format(cost))
print('Expected cost (approx): 2.13')
print('Gradient at test theta - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.3460\n 0.0851\n 0.1185\n 0.1506\n 0.0159')

'''第2部分 尝试不同的惩罚系数[0,1,10,100],分别利用高级优化算法求解最优参数，分别计算训练好的分类器在训练集上的准确率，
并画出决策边界
 '''
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lmd = 1  # 需要改变这个值
"""lmd 正则化惩罚可以防止过拟合"""
# Optimize
def cost_func(t):
    return cost_function_reg(t, X, y, lmd)[0]


def grad_func(t):
    return cost_function_reg(t, X, y, lmd)[1]


theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=initial_theta, maxiter=400, full_output=True,
                                     disp=False)

print('Plotting decision boundary ...')
plot_decision_boundary(theta, X, y)
plt.show()
input('Program paused. Press ENTER to continue')
plt.title('lambda = {}'.format(lmd))

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

p = predict(theta, X)

print('Train Accuracy: {:0.4f}'.format(np.mean(y == p) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

