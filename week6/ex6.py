import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import linearRegCostFunction as lrcf
import trainLinearReg as tlr
import learningCurve as lc
import polyFeatures as pf
import featureNormalize as fn
import plotFit as plotft
import validationCurve as vc

'''第1部分 加载并可视化数据集'''

print('Loading and Visualizing data ...')

data = scio.loadmat('ex5data1.mat')  # 读取矩阵格式的数据集
# 数据集已经被分成了训练集、验证集、测试集三部分
X = data['X']  # 提取训练集原始输入特征
y = data['y'].flatten()  # 提取训练集输出变量 并转换成一维数组
print(y.shape)
Xval = data['Xval']  # 提取验证集原始输入特征
yval = data['yval'].flatten()  # 提取验证集输出变量 并转换成一维数组
Xtest = data['Xtest']  # 提取测试集原始输入特征
ytest = data['ytest'].flatten()  # 提取测试集输出变量 并转换成一维数组
m = y.size  # 训练样本数

# 可视化训练集
plt.figure()
plt.scatter(X, y, c='r', marker="x")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water folowing out of the dam (y)')

input('Program paused. Press ENTER to continue')

'''第2-1部分 编写正则化线性回归的代价函数'''

theta = np.ones(2)  # 初始化参数为1  只有一个原始输入特征 所以两个参数
cost, _ = lrcf.linear_reg_cost_function(theta, np.c_[np.ones(m), X], y, 1)  # 为原始输入特征矩阵前面加一列1 正则化系数为1

# 返回计算的代价并与期望进行比较 验证程序正确性
print('Cost at theta = [1  1]: {:0.6f}\n(this value should be about 303.993192'.format(cost))



'''第2-2部分 计算正则化线性回归的梯度'''

theta = np.ones(2)  # 初始化参数为1  只有一个原始输入特征 所以两个参数
cost, grad = lrcf.linear_reg_cost_function(theta, np.c_[np.ones(m), X], y, 1)  # 为原始输入特征矩阵前面加一列1 正则化系数为1

# 返回计算的代价和梯度，并将梯度与期望进行比较  验证程序正确性
print('Gradient at theta = [1  1]: {}\n(this value should be about [-15.303016  598.250744]'.format(grad))

'''第3部分 训练线性回归'''
lmd = 0  # 相当于不使用正则化

theta = tlr.train_linear_reg(np.c_[np.ones(m), X], y, lmd)  # 返回训练后的最优参数

# 画出拟合的曲线
plt.plot(X, np.dot(np.c_[np.ones(m), X], theta))
plt.show()
input('Program paused. Press ENTER to continue')
"""由于原始输入特征只有1个，所以模型的拟合效果不是很好(underfitting)，之后我们在原始输入特征的基础上增加多项式特征。"""
'''第4部分 绘制线性回归学习曲线'''
lmd = 0  # 相当于不使用正则化
# 返回不同训练样本下的训练误差和验证误差
error_train, error_val = lc.learning_curve(np.c_[np.ones(m), X], y, np.c_[np.ones(Xval.shape[0]), Xval], yval, lmd)

# 绘制学习曲线
plt.figure()
plt.plot(np.arange(m), error_train, np.arange(m), error_val)
plt.title('Learning Curve for Linear Regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()
input('Program paused. Press ENTER to continue')

"""high bias, underfitting, adding more feature or polynomial features"""

"""  hθ (x) = θ0 + θ1 ∗ (waterLevel) + θ2 ∗ (waterLevel)2 + · · · + θp ∗ (waterLevel)p= θ0 +θ1x1 +θ2x2 +...+θpxp.      """

'''第5部分 增加多项式特征'''

p = 5  # 多项式的最高次数

# 分别对训练集、验证集、测试集的原始输入特征矩阵增加新的多项式特征，返回新的输入特征矩阵 再加一列特征1 方便矩阵运算
# 并对新的输入特征矩阵进行特征缩放 使各个特征的取值范围相近 加快优化速度
# 验证集和测试集特征缩放使用的均值和方差 使用训练集计算的均值和方差

X_poly = pf.poly_features(X, p)
X_poly, mu, sigma = fn.feature_normalize(X_poly)
X_poly = np.c_[np.ones(m), X_poly]

X_poly_test = pf.poly_features(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), X_poly_test]

X_poly_val = pf.poly_features(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), X_poly_val]

print('Normalized Training Example 1 : \n{}'.format(X_poly[0]))

'''第6部分 增加多项式特征后进行训练  可视化拟合效果和学习曲线'''

lmd = 0  # 不进行正则化
theta = tlr.train_linear_reg(X_poly, y, lmd)  # 训练得到最优参数

# 可视化训练集和拟合曲线
plt.figure()
plt.scatter(X, y, c='r', marker="x")
plotft.plot_fit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water folowing out of the dam (y)')
plt.ylim([0, 60])
plt.title('Polynomial Regression Fit (lambda = {})'.format(lmd))
plt.show()
input('Program paused. Press ENTER to continue')
# 绘制学习曲线
error_train, error_val = lc.learning_curve(X_poly, y, X_poly_val, yval, lmd)
plt.figure()
plt.plot(np.arange(m), error_train, np.arange(m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(lmd))
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()
input('Program paused. Press ENTER to continue')
print('Polynomial Regression (lambda = {})'.format(lmd))
print('# Training Examples\tTrain Error\t\tCross Validation Error')
for i in range(m):
    print('  \t{}\t\t{}\t{}'.format(i, error_train[i], error_val[i]))

"""增加polynomial后，traning error一直很低，但是cv误差不是很好，说明overfitting,可以通过regularization来解决，选择一个合适的lambda"""
'''第7部分 通过验证集选择一个最优的lambda值，模型选择'''

lambda_vec, error_train, error_val = vc.validation_curve(X_poly, y, X_poly_val, yval)

plt.figure()
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
print('验证误差')
print(error_val)
print('使验证误差最小的lambda取值:')
print(lambda_vec[np.argmin(error_val)])  # 使验证误差最小的lambda取值
print("chage line108 lmd=0 into lmd=1")
plt.show()