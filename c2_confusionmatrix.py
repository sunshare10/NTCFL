import numpy as np
import matplotlib.pyplot as plt
from get_data import get_data2
import os
import sys
# from typing import Sequence
sys.path.insert(0,os.getcwd())
import copy
import argparse
import shutil
import time
import datetime
import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from get_data import get_data2
from utils.history import History
from utils.dataloader import Mydataset, collate
from torch.utils.data import Dataset,TensorDataset
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
from core.optimizers import *
from models.build import BuildNet
class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, labels, predicts):
        """
		:param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)        
        :return:
        """
        for predict, label in zip(labels, predicts):
            self.matrix[label, predict] += 1

    def getMatrix(self,normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # 百分比转换
            self.matrix=np.around(self.matrix, 2)   # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict('models/swin_transformer/base_160.py')
    model = BuildNet(model_cfg)
    # if device != torch.device('cpu'):
    #     model = DataParallel(model,device_ids=[0])
    model = init_model(model, data_cfg, device=device, mode='eval')
    with open('markdict.txt','r+') as fr:
        classdict=eval(fr.read())
    classes_names,indexs=[],[]
    labels_name=list(classdict.keys())
    indexs=list(classdict.values())
    # labels_name=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)  # 实例化
    NUM_HEADERS=16
    PCK_LEN=1500
    PNG_chan=160  #灰度图尺寸
    niming_flag=1  #0表示不匿名，1表示匿名不带方向，2表示匿名且标识方向,3表示用255进行匿名，不带方向
    t=1
    num_users=4
    x_train, y_train,x_test, y_test,train_dataset,test_dataset,train_loader,val_loader ={},{},{},{},{},{},{},{}
    for idx in range(num_users):
        data = np.load('D:/datasets/federated/4客户端24小时12类/client'+'_'+str(idx)+'_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+'.npz')
        x_train[idx], x_test[idx], y_train[idx], y_test[idx] = data['x_train'],data['x_test'],data['y_train'],data['y_test']
        # print(x_train[idx], x_test[idx], y_train[idx], y_test[idx])
        if NUM_HEADERS==16 and PCK_LEN==54:
            x_train[idx]=x_train[idx].reshape(-1,1,60,60)
            x_test[idx]=x_test[idx].reshape(-1,1,60,60)
        elif NUM_HEADERS==16 and PCK_LEN==72:
            x_train[idx]=x_train[idx].reshape(-1,1,60,60)
            x_test[idx]=x_test[idx].reshape(-1,1,60,60)
        elif NUM_HEADERS==8 and PCK_LEN==1500:
            x_train[idx]=x_train[idx].reshape(-1,1,112,112)
            x_test[idx]=x_test[idx].reshape(-1,1,112,112)
        elif NUM_HEADERS==16 and PCK_LEN==1500:
            x_train[idx]=x_train[idx].reshape(-1,1,160,160)
            x_test[idx]=x_test[idx].reshape(-1,1,160,160)
        elif NUM_HEADERS==24 and PCK_LEN==1500:
            x_train[idx]=x_train[idx].reshape(-1,1,192,192)
            x_test[idx]=x_test[idx].reshape(-1,1,192,192)
        elif NUM_HEADERS==32 and PCK_LEN==1500:
            x_train[idx]=x_train[idx].reshape(-1,1,224,224)
            x_test[idx]=x_test[idx].reshape(-1,1,224,224)
        elif NUM_HEADERS==40 and PCK_LEN==1500:
            x_train[idx]=x_train[idx].reshape(-1,1,280,280)
            x_test[idx]=x_test[idx].reshape(-1,1,280,280)
        elif NUM_HEADERS==48 and PCK_LEN==1500:
            x_train[idx]=x_train[idx].reshape(-1,1,288,288)
            x_test[idx]=x_test[idx].reshape(-1,1,288,288)
        if idx==0:
            x_test_all=x_test[idx]
            y_test_all=y_test[idx]
        else:
            x_test_all= np.concatenate((x_test_all, x_test[idx]), axis=0)
            y_test_all= np.concatenate((y_test_all, y_test[idx]), axis=0)
        x_train[idx]=torch.from_numpy(x_train[idx])
        y_train[idx]=torch.from_numpy(y_train[idx])
        x_test[idx]=torch.from_numpy(x_test[idx])
        y_test[idx]=torch.from_numpy(y_test[idx])
        x_train[idx]=x_train[idx].type(torch.FloatTensor)
        y_train[idx]=y_train[idx].type(torch.LongTensor)
        x_test[idx]=x_test[idx].type(torch.FloatTensor)
        y_test[idx]=y_test[idx].type(torch.LongTensor)
        #封装数据
        # print(x_train[idx][0])
        train_dataset[idx]=TensorDataset(x_train[idx],y_train[idx])
        test_dataset[idx]=TensorDataset(x_test[idx],y_test[idx])
    print(x_test_all.shape)
    print(y_test_all.shape)
    x_test_all=torch.from_numpy(x_test_all)
    y_test_all=torch.from_numpy(y_test_all)
    x_test_all=x_test_all.type(torch.FloatTensor)
    y_test_all=y_test_all.type(torch.LongTensor)
    test_dataset_all=TensorDataset(x_test_all,y_test_all)
    test_loader = DataLoader(test_dataset_all,  shuffle=False, batch_size=16, num_workers=data_cfg.get('num_workers'), pin_memory=True,
        drop_last=True)
    
    with torch.no_grad():
        for index, (labels, imgs) in enumerate(test_loader):
            imgs=imgs.to(device)
            
            labels_pd = model(imgs,return_loss=False)
            predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)   # array([0,5,1,6,3,...],dtype=int64)
            labels_np = labels.numpy()                                          # array([0,5,0,6,2,...],dtype=int64)
            drawconfusionmatrix.update(labels_np, predict_np)   # 将新批次的predict和label更新（保存）
        
    drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵

    confusion_mat=drawconfusionmatrix.getMatrix() # 你也可以使用该函数获取混淆矩阵(ndarray)
    print(confusion_mat)
