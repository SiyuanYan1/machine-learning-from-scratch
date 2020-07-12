import numpy as np
from sigmoid import *
# theta1:25*401 输入层多一个偏置项
# theta2:10*26  隐藏层多一个偏置项
def predict(theta1, theta2, x):
    m=x.shape[0]
    num_label=theta2.shape[0]
    x=np.c_[np.ones(m),x] # 5000*401
    p=np.zeros(m)
    z2=x.dot(theta1.T) # 5000*401
    a2=sigmoid(z2)  # 5000*25
    a2=np.c_[np.ones(m),a2] #5000*26
    z3=a2.dot(theta2.T)  # 5000*10
    a3=sigmoid(z3)
    p=np.argmax(a3,axis=1)
    p+=1
    return p



