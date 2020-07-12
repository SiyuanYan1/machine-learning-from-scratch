import pandas as pd
import numpy as np
import scipy as sp

data=pd.read_csv('Pokemon.csv')

X=data.loc[:560,['Total','HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
y=np.array(data.loc[:560,'Attack'])

X['constant']=1 #bias
# print(X)

# #梯度下降找parameter使loss最小
def sgd(X ,y_true ,w ,lr=0.1 ,iteration=10):
    for k in range(iteration):
        sum_loss = 0
        for i in range(len(X)):  # len(X)=row of X

            loss=np.dot(X.iloc[i,:],w)-y_true[i]  #sum(X.iloc[i,:]*w)
            sum_loss+=loss**2
            for j in range(X.shape[1]):#shape1 =columns
                grad=2*loss*X.iloc[i,j]
                w[j]-= (1/X.shape[0]) * lr *grad
        print('iteration: '+str(k)+' weight:'+str(w)+ 'loss: '+str(sum_loss))
    return w

#最优解
def adagrad(X ,y_true ,w ,lr=0.1 ,iteration=10):
    s_grad=np.zeros(X.shape[1])
    for i in range(iteration):
        hypothesis=np.dot(X,w)
        loss=hypothesis-y_true
        s_loss = np.mean(np.square(loss))
        grad=np.dot(X.transpose(),loss)*(-2)   #得到一个w*1的矩阵，每个w一个对应的grad
        s_grad+=grad**2
        ada=np.sqrt(s_grad)
        w-=(lr/ada) * grad
        print('iteration: '+str(i)+' weight: '+str(w)+ 'loss: '+str(s_loss))
    return w



w=np.zeros(X.shape[1])

# print(w)
# print(w)
adagrad(X,y,w,10,100)







# def adagrad1(X,y_true,w,lr=0.1,iteration=10):
#     for i in range(iteration):
#         sum_loss=0
#         grad=np.array([0.,0.,0.,0.,0.,0.,0.])
#         for i in range(len(X)):
#             loss = np.dot(X.iloc[i, :], w) - y_true[i]  # sum(X.iloc[i,:]*w)
#             sum_loss += loss ** 2













