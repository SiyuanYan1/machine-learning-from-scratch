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
def map_feature(x1, x2):
    degree = 6

    x1 = x1.reshape((x1.size, 1))
    x2 = x2.reshape((x2.size, 1))
    result = np.ones(x1[:, 0].shape)

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            result = np.c_[result, (x1 ** (i - j)) * (x2 ** j)]

    return result

def plot_data(X,y):
    plt.figure(0)
    positive=X[y==1]
    negative=X[y==0]
    plt.scatter(positive[:,0],positive[:,1],marker='+',c='red',label='Admitted')
    plt.scatter(negative[:,0],negative[:,1],marker='o',c='blue',label='Not Admitted')
    plt.xlabel('Exam1 score')
    plt.ylabel('Exam2 score')
    plt.legend(['Admitted','Not Admitted'],loc='best')
    # plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
def h(theta,x):
    return sigmoid(x.dot(theta))
def compute_cost(x,y,theta):
    m=y.size
    term1=y.dot(np.log(h(theta,x)))
    term2=(1-y).dot(np.log(1-h(theta,x)))
    cost=(-1/m) * (term1+term2)
    grad=( (h(theta,x)-y).dot(x) )/m
    return cost,grad

def predict(theta,X):

    p=sigmoid(X.dot(theta))
    p[p>=0.5]=1
    p[p<0.5]=0
    return p

#Part 1 Visualizing the data
data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,0:2]
y=data[:,2]
print('Plotting Data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
plot_data(X,y)

#Part2 Cost function and gradient
(m, n) = X.shape  # m样本数 n原始输入特征数
X = np.c_[np.ones(m), X]  # 特征矩阵X前加一列1  方便矩阵运算

# 初始化模型参数为0
initial_theta = np.zeros(n + 1)

# 计算逻辑回归的代价函数和梯度
cost, grad = compute_cost( X, y,initial_theta)

np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})  # 设置输出格式

# 与期望值进行比较 验证程序的正确性
print('Cost at initial theta (zeros): {:0.3f}'.format(cost))  # 0参数下的代价函数值
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n{}'.format(grad))  # 0参数下的梯度值
print('Expected gradients (approx): \n-0.1000\n-12.0092\n-11.2628')

# 用非零参数值计算代价函数和梯度
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = compute_cost( X, y,test_theta)
# 与期望值进行比较 验证程序的正确性
print('Cost at test theta (zeros): {}'.format(cost))
# 非0参数下的代价函数值
print('Expected cost (approx): 0.218')
print('Gradient at test theta: \n{}'.format(grad))
print('Expected gradients (approx): \n0.043\n2.566\n2.647')  # 非0参数下的代价函数值

# Part 3
'''第3部分 用高级优化方法fmin_bfgs求解最优参数'''
#可以把高级优化想像成梯度下降法 只不过不用人工设置学习率
'''
    fmin_bfgs优化函数 第一个参数是计算代价的函数 第二个参数是计算梯度的函数 参数x0传入初始化的theta值
    maxiter设置最大迭代优化次数
'''
def cost_function(theta):
    return compute_cost(X,y,theta)[0]#返回cost
def grad_function(theta):
    return compute_cost(X, y, theta)[1]#返回梯度

# 运行高级优化方法
theta, cost, *unused = opt.fmin_bfgs(f=cost_function, fprime=grad_function, x0=initial_theta, maxiter=400, full_output=True, disp=False)
# 打印最优的代价函数值和参数值  与期望值比较 验证正确性
print('Cost at theta found by fmin: {:0.4f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: \n{}'.format(theta))
print('Expected Theta (approx): \n-25.161\n0.206\n0.201')

# 画出决策边界
plot_decision_boundary(theta, X, y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')


'''第4部分 用训练好的分类器进行预测，并计算分类器在训练集上的准确率'''
#假设一个学生 考试1成绩45 考试2成绩85  预测他通过的概率
prob = sigmoid(np.array([1, 45, 85]).dot(theta))
# 与期望值进行比较 验证正确性
print('For a student with scores 45 and 85, we predict an admission probability of {:0.4f}'.format(prob))
print('Expected value : 0.775 +/- 0.002')

# 计算分类器在训练集上的准确率
p = predict(theta, X)
# 与期望值进行比较 验证正确性
print('Train accuracy: {}'.format(np.mean(y == p) * 100))
print('Expected accuracy (approx): 89.0')
plt.show()







