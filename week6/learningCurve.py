import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf
def learning_curve(X,y,Xval,yval,lmd):
    m=X.shape[0]
    error_train=np.zeros(m)
    error_val=np.zeros(m)

    for i in range(m):
        x=X[:i+1,:]
        y1=y[:i+1]
        theta=tlr.train_linear_reg(x,y1,lmd)
        error_train[i]=lrcf.linear_reg_cost_function(theta,x,y1,lmd)[0]
        error_val[i]=lrcf.linear_reg_cost_function(theta,Xval,yval,lmd)[0]
    return error_train,error_val




