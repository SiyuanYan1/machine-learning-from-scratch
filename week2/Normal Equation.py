import numpy as np

'''第3部分 正规方程法求解多元线性回归'''
#当n小于10000， normal equation更好， 如果n过大，矩阵乘法将变得很慢
# 正规方程法不用进行特征缩放
#****在numpy中，一个列表虽然是横着表示的，但它是列向量。
def normal_equation(X,y):
    #theta=np.zeros( (X.shape[1],1) )
    theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


print('Solving with normal equations ...')

data=np.loadtxt('ex1data2.txt',delimiter=',',dtype=np.int64)
X=data[:,0:2]
y=data[:,2] #m*1
m=y.size
X=np.c_[np.ones(m),X]  #m*(n+1)
theta=normal_equation(X,y)
# 打印求解的最优参数
print('Theta computed from the normal equations : \n{}'.format(theta))

x2=np.array([1,1650,3])
price=x2.dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : {:0.3f}'.format(price))

