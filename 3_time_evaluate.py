import torch
import datetime
from get_data import get_data2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,TensorDataset

x_train, y_train,x_test, y_test=get_data2()
#转换numpy为tensor
x_train=x_train.reshape(56638,1,16,54)
x_test=x_test.reshape(6294,1,16,54)
x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)
x_test=torch.from_numpy(x_test)
y_test=torch.from_numpy(y_test)
#修改数据格式
x_train=x_train.type(torch.FloatTensor)
y_train=y_train.type(torch.LongTensor)
x_test=x_test.type(torch.FloatTensor)
y_test=y_test.type(torch.LongTensor)
#封装数据
# print(x_train[0])
train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)
testloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
timeseries=[]
timeseries.append(0)
oldtime=datetime.datetime.now()
global_model=torch.load('./global_model')
# global_model.to(device)
# global_model.train()
with torch.no_grad():
    correct = 0
    total = 0
    a=[]
    cishu=0
    for images, labels in testloader:
        images = images.to('cpu')
        labels = labels.to('cpu')
        outputs = global_model(images)
        _, predicted = torch.max(outputs.data, 1)#预测结果
        total += labels.size(0)#正确结果
        correct += (predicted == labels).sum().item() #正确结果总数
        newtime=datetime.datetime.now()
        feng=(newtime-oldtime).seconds
        miao=(newtime-oldtime).microseconds
        a=feng+miao/1000000
        cishu=cishu+1
        if(cishu==1 or cishu==10 or cishu==100):
            timeseries.append(a)
    print(timeseries)
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
import numpy as np
import matplotlib.pyplot as plt
plt.figure()
x=[i*100 for i in range(1,len(timeseries)+1)]
plt.plot(x, timeseries)
plt.xlabel('number')
plt.ylabel('microseconds')
plt.show()