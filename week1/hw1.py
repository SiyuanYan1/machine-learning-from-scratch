import pandas as pd
import numpy as np

import math

data = pd.read_csv('train.csv')
###取出所有pm2.5数据
pm2_5=data[data['observations']=='PM2.5'].iloc[:,3:]
# print(pm2_5)

tempxlist=[]
tempylist=[]

#i=24-10+1=15

for i in range(15):
    tempx=pm2_5.iloc[:,i:i+9]
    tempy=pm2_5.iloc[:,i+9]
    tempx.columns=np.array(range(9))
    tempy.columns=['1']
    tempxlist.append(tempx)
    tempylist.append(tempy)
#feature数据

xdata=pd.concat(tempxlist)
x=np.array(xdata,float)

# print(x.shape)
#label数据
ydata=pd.concat(tempylist)
y=np.array(ydata,float)
# print(y)
# print(y.shape)

#训练模型
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)  #在feature基础上加入bias
w=np.zeros((len(x[0])))
lr=10
iteration=100
s_grad=np.zeros(len(x[0]))
# print(x)
x_t=x.transpose()


for i in range(iteration):
    hypothesis=np.dot(x,w)
    loss=y-hypothesis
    grad=np.dot(x_t,loss)*(-2)
    s_grad+=grad**2
    s_loss = np.mean(np.square(loss))
    ada=np.sqrt(s_grad)
    w=w-(lr/ada)*grad
    print('iteration: ' + str(i) + ' weight: ' + str(w) + 'loss: ' + str(s_loss))

print(w)



#测试
test_data=pd.read_csv('test.csv')
pm2_5_test=test_data[test_data['AMB_TEMP']=='PM2.5'].iloc[:,2:]
# print(pm2_5_test)
x_test=np.array(pm2_5_test,float)
x_test_b=np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)#加入bias
y_star=np.dot(x_test_b,w)
# print(y_star)
y_pre=pd.read_csv("sampleSubmission.csv")
y_pre.value=y_star




