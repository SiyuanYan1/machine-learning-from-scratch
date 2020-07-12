import scipy.optimize as opt #高级优化函数的包
import lrCostFunction as lCF
from sigmoid import *


#定义一个优化函数 实际上调用的是Python内置的高级优化函数 
#可以把它想象成梯度下降法 但是不用手动设置学习率
''' fmin_cg优化函数 第一个参数是计算代价的函数 第二个参数是计算梯度的函数 参数x0传入初始化的theta值
    args传入训练样本的输入特征矩阵X,对应的2分类新标签y,正则化惩罚项系数lmd
    maxiter设置最大迭代优化次数
'''
def optimizeTheta(theta,X,y,lmd):
    res=opt.fmin_cg(lCF.Compute_cost,fprime=lCF.Compute_grad,x0=theta,\
                    args=(X,y,lmd),maxiter=50,disp=False,full_output=True)
    return res[0],res[1]

def one_vs_all(X, y, num_labels, lmd):
    
    (m, n) = X.shape #m为训练样本数  n为原始输入特征数

    '''
    逻辑回归多分类器的训练过程：
    用逻辑回归做多分类 相当于做多次2分类 每一次把其中一个类别当作正类 其余全是负类
    手写数字识别是10分类 需要做十次2分类 
    比如：第一次把数字0当作正类 设置新的标签为1  数字1-9为负类  设置新的标签是0 进行2分类
         第一次把数字1当作正类 设置新的标签为1  数字2-9和0为负类  设置新的标签是0 进行2分类
         以此类推

    '''
    all_theta = np.zeros((num_labels, n + 1)) #存放十次2分类的 最优化参数
    initial_theta=np.zeros(n+1)   #每一次2分类的初始化参数值

    X = np.c_[np.ones(m), X]  #添加一列特征 值为1


    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))
        iclass=i if i else 10 #数字0 属于第十个类别
        logic_Y=np.array([1 if x==iclass else 0 for x in y]) #设置每次2分类的新标签
        itheta,imincost=optimizeTheta(initial_theta,X,logic_Y,lmd)
        all_theta[i,:]=itheta
    print('Done')

    return all_theta #返回十次2分类的 最优化参数
