import numpy as np


def recover_data(Z, U, K):  #进行压缩重放
    
    X_rec = np.zeros((Z.shape[0], U.shape[0])) #原始样本在特征向量上的投影点 X_rec:m*n Z:m*K  U:n*n

    X_rec=Z.dot(U[:,:K].T)

    return X_rec
