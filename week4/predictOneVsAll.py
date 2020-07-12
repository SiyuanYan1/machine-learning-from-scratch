import numpy as np
import lrCostFunction as lCF


def predict_one_vs_all(all_theta, X):
    m = X.shape[0]   #shape[0]返回2维数组的行数   样本数
  
    p = np.zeros(m)  #存储分类器预测的类别标签

    X = np.c_[np.ones(m), X]  #增加一列1 X:5000*401

    Y=lCF.h(X,all_theta.T)  #all_theta:10*401   Y:5000*10  每一行是每一个样本属于10个类别的概率

    p=np.argmax(Y,axis=1) #找出每一行最大概率所在的位置
    
    p[p==0]=10  #如果是数字0的话  他属于的类别应该是10

    return p
