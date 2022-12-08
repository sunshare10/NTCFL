import imp
import os
import sys
sys.path.insert(0,os.getcwd())
import argparse
import numpy as np
import copy
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable
from get_data import get_data3
from torch.utils.data import Dataset,TensorDataset
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv
from sklearn.model_selection import train_test_split
from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model
import matplotlib.pyplot as plt


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
def train_val_test(dataset, idxs):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    args = parse_args()
    # split indexes for train and test (80, 20)
    idxs_train = idxs[:int(0.8*len(idxs))]
    idxs_test = idxs[int(0.8*len(idxs)):]

    trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=args.local_bs, shuffle=True)
    testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                            batch_size=args.local_bs, shuffle=False)
    #return trainloader, validloader, testloader
    return trainloader, testloader
def get_metrics_output(eval_results, metrics_output,classes_names, indexs):
    f = open(metrics_output,'a', newline='')
    writer = csv.writer(f)
    # classes_names=20 #共20类
    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    p_r_f1 = [['Classes','Precision','Recall','F1 Score']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        p_r_f1.append(data)
    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    writer.writerows(TABLE_DATA_1)
    writer.writerow([])
    print()

    TITLE = 'Total Results'    
    TABLE_DATA_2 = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1',0.0)), '{:.2f}'.format(eval_results.get('accuracy_top-5',100.0)), '{:.2f}'.format(mean(eval_results.get('precision',0.0))),'{:.2f}'.format(mean(eval_results.get('recall',0.0))),'{:.2f}'.format(mean(eval_results.get('f1_score',0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    writer.writerows(TABLE_DATA_2)
    writer.writerow([])
    print()


    writer_list     = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    for i in range(len(classes_names)):
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    TITLE = 'Confusion Matrix'
    TABLE_DATA_3 = tuple(writer_list)
    table_instance = AsciiTable(TABLE_DATA_3,TITLE)
    print(table_instance.table)
    writer.writerows(TABLE_DATA_3)
    print()

def get_prediction_output(preds,targets,image_paths,classes_names,indexs,prediction_output):
    nums = len(preds)
    f = open(prediction_output,'a', newline='')
    writer = csv.writer(f)
    
    results = [['File', 'Pre_label', 'True_label', 'Success']]
    results[0].extend(classes_names)
    
    for i in range(nums):
        temp = [image_paths[1]]
        pred_label = classes_names[indexs[torch.argmax(preds[i]).item()]]
        true_label = classes_names[indexs[targets[i].item()]]
        success = True if pred_label == true_label else False
        class_score = preds[i].tolist()
        temp.extend([pred_label,true_label,success])
        temp.extend(class_score)
        results.append(temp)
        
    writer.writerows(results)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C,参加训练的客户端比例')
    parser.add_argument('--num_users', type=int, default=4,
                        help="number of users: K")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
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
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix)  # 仅画出颜色格子，没有值
        # plt.title("Normalized confusion matrix")  # title
        plt.xlabel("真实标签")
        plt.ylabel("预测标签")
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

def main(): 
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    with open('markdict.txt','r+') as fr:
        classdict=eval(fr.read())
    classes_names,indexs=[],[]
    labels_name=list(classdict.keys())
    indexs=list(classdict.values())
    # labels_name=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)  # 实例化
    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    NUM_HEADERS=16
    PCK_LEN=1500
    PNG_chan=160  #灰度图尺寸
    niming_flag=1  #0表示不匿名，1表示匿名不带方向，2表示匿名且标识方向,3表示用255进行匿名，不带方向
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t=1
    # 初始化
    meta = dict()
    dirname = str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    save_dir = os.path.join('eval_results',model_cfg.get('backbone').get('type'),dirname)
    os.makedirs(save_dir)
    metrics_output = os.path.join(save_dir,'metrics_output.csv')
    prediction_output = os.path.join(save_dir,'prediction_results.csv')
    
    """
    获取类别名以及对应索引、获取标注文件
    """
    # classes_map = 'datas/annotations.txt' 
    # test_annotations    = 'datas/test.txt'
    # classes_names, indexs = get_info(classes_map)
    # with open(test_annotations, encoding='utf-8') as f:
    #     test_datas   = f.readlines()
        
    """
    生成模型、加载权重
    """
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)

    model = init_model(model, data_cfg, device=device, mode='eval')

    x_train, y_train,x_test, y_test,train_dataset,test_dataset,train_loader,val_loader ={},{},{},{},{},{},{},{}
    for idx in range(args.num_users):
        #4客户端24小时12类  11类
        data = np.load('D:/datasets/federated/4客户端24小时12类/client'+'_'+str(idx)+'_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+'.npz')
        # data = np.load('E:/Awesome-Backbones-main/client'+'_'+str(idx)+'_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+'.npz')
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
        train_loader[idx] = DataLoader(train_dataset[idx],  shuffle=False, batch_size=96, num_workers=data_cfg.get('num_workers'), pin_memory=True,
        drop_last=True)
        val_loader[idx] = DataLoader(test_dataset[idx],  shuffle=False, batch_size=96, num_workers=data_cfg.get('num_workers'), pin_memory=True,
        drop_last=True)
    x_test_all=torch.from_numpy(x_test_all)
    y_test_all=torch.from_numpy(y_test_all)
    x_test_all=x_test_all.type(torch.FloatTensor)
    y_test_all=y_test_all.type(torch.LongTensor)
    test_dataset_all=TensorDataset(x_test_all,y_test_all)
    test_loader = DataLoader(test_dataset_all,  shuffle=False, batch_size=args.local_bs, num_workers=data_cfg.get('num_workers'), pin_memory=True,
        drop_last=True)

    with open('markdict.txt','r+') as fr:
        classdict=eval(fr.read())
    classes_names,indexs=[],[]
    classes_names=list(classdict.keys())
    indexs=list(classdict.values())
    # classes_names=[{'BaiduNetdisk': 0, 'Chrome': 1, 'Cloudmusic': 2, 'Firefox': 3, 'HUYA': 4, 'IQIYI': 5, 'MicrosoftEdge': 6, 'QQ': 7, 'QQMusic': 8, 'Wechat': 9, 'Wemeetapp': 10, 'YY': 11}]
    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    with torch.no_grad():
        preds,targets, image_paths = [],[],[]
        with tqdm(total=len(test_dataset)//args.local_bs) as pbar:
            for _, batch in enumerate(test_loader):
                images, target = batch
                outputs = model(images.to(device),return_loss=False)
                predict_np = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
                labels_np = target.numpy() 
                drawconfusionmatrix.update(labels_np, predict_np)
                preds.append(outputs)
                targets.append(target.to(device))
                image_paths.extend("image_path")
                pbar.update(1)
    drawconfusionmatrix.drawMatrix()
    confusion_mat=drawconfusionmatrix.getMatrix() # 你也可以使用该函数获取混淆矩阵(ndarray)
    print(confusion_mat)            
    eval_results = evaluate(torch.cat(preds),torch.cat(targets),data_cfg.get('test').get('metrics'),data_cfg.get('test').get('metric_options'))
    get_metrics_output(eval_results,metrics_output,classes_names,indexs)
    get_prediction_output(torch.cat(preds),torch.cat(targets),image_paths, classes_names, indexs, prediction_output)                 

if __name__ == "__main__":
    main()
