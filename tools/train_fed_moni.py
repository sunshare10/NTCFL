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
import numpy as np
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from get_data import get_data3
from utils.history import History
from utils.dataloader import Mydataset, collate
from torch.utils.data import Dataset,TensorDataset
from utils.train_utils import  trainfed, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model, validationfed
from utils.inference import init_model
from core.optimizers import *
from models.build import BuildNet
from sklearn.model_selection import train_test_split
# from sampling import mnist_iid


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
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
def main():
    # 读取配置文件获取关键字段
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    print_info(model_cfg)
    NUM_HEADERS=16
    PCK_LEN=1500
    PNG_chan=160  #灰度图尺寸
    niming_flag=1  #0表示不匿名，1表示匿名不带方向，2表示匿名且标识方向,3表示用255进行匿名，不带方向
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t=10
    # 初始化
    meta = dict()
    dirname = str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('logs',model_cfg.get('backbone').get('type'),dirname)
    meta['save_dir'] = save_dir
    meta['num_users']=args.num_users
    
    # 设置随机数种子
    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=args.deterministic)
    meta['seed'] = seed
    
    # 初始化模型,详见https://www.bilibili.com/video/BV12a411772h
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(str(device))
    print('Initialize the weights.')
    model = BuildNet(model_cfg)
    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))
    # print(model)
    if device != torch.device('cpu'):
        model = DataParallel(model,device_ids=[args.gpu_id])
    
    # 初始化优化器
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(),**optimizer_cfg)
    
    # 初始化学习率更新策略
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)
    dirs1={}
    # dirs1[0]=['/hy-tmp/client0/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/','/hy-tmp/client0/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming2/']
    # dirs1[1]=['/hy-tmp/client1/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[2]=['/hy-tmp/client2/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[3]=['/hy-tmp/client3/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'ipniming/']
    dirs1[0]=['e:/1/client1/07050021/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/',
              'e:/1/client1/07100002/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    dirs1[1]=['e:/1/client2/07101940/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    dirs1[2]=['e:/1/client3/07110005/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    dirs1[3]=['e:/1/last/version819/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'ipniming/']
    x_train, y_train,x_test, y_test,train_dataset,test_dataset,train_loader,val_loader ={},{},{},{},{},{},{},{}
    x_std,y={},{}
    for idx in range(args.num_users):
        x_std[idx], y[idx]=get_data3(num_headers = NUM_HEADERS,pck_len=PCK_LEN,pngchang=PNG_chan,
                                              t=t,niming_flag=niming_flag,dirs=dirs1[idx])
        if idx==0:
            y_all=y[idx]
        else:
            y_all= np.concatenate((y_all, y[idx]), axis=0)
    classnames=np.unique(y_all)
    markdict={}
    i=0
    for classname in classnames:
        markdict[classname]=int(i)
        i=i+1
    print(markdict)
    for idx in range(args.num_users):
    # ros = RandomOverSampler(random_state=0)
    # x_train, y_train = ros.fit_resample(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    #转换numpy为tensor
        for i in range(len(y[idx])):
            y[idx][i]=markdict[y[idx][i]]
        y[idx]=y[idx].astype(int)
        # print(type(y[idx]))
        
        x_train[idx], x_test[idx], y_train[idx], y_test[idx] = train_test_split(x_std[idx], y[idx], test_size=0.2, random_state=False)
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
        train_loader[idx] = DataLoader(train_dataset[idx],  shuffle=False, batch_size=16, num_workers=data_cfg.get('num_workers'), pin_memory=True,
        drop_last=True)
        val_loader[idx] = DataLoader(test_dataset[idx],  shuffle=False, batch_size=16, num_workers=data_cfg.get('num_workers'), pin_memory=True,
        drop_last=True)
    x_test_all=torch.from_numpy(x_test_all)
    y_test_all=torch.from_numpy(y_test_all)
    x_test_all=x_test_all.type(torch.FloatTensor)
    y_test_all=y_test_all.type(torch.LongTensor)
    test_dataset_all=TensorDataset(x_test_all,y_test_all)
    val_loader_all = DataLoader(test_dataset_all,  shuffle=False, batch_size=16, num_workers=data_cfg.get('num_workers'), pin_memory=True,
        drop_last=True)
    # 将关键字段存储，方便训练时同步调用&更新
    runner = dict(
        optimizer         = optimizer,
        train_loader      = train_loader,
        val_loader        = val_loader_all,
        iter              = 0,
        epoch             = 0,
        max_epochs       = data_cfg.get('train').get('epoches'),
        max_iters         = data_cfg.get('train').get('epoches')*len(train_loader),
        best_train_loss   = float('INF'),
        best_val_acc     = float(0),
        best_train_weight = '',
        best_val_weight   = '',
        last_weight       = ''
    )
    meta['train_info'] = dict(train_loss = [],
                            val_loss = [],
                            train_acc = [],
                            val_acc = [])
    for i in range(args.num_users):
        meta['train_info'+str(i)] = dict(train_loss = [],
                                    val_loss = [],
                                    train_acc = [],
                                    val_acc = [])
    # 是否从中断处恢复训练
    if args.resume_from:
        model,runner,meta = resume_model(model,runner,args.resume_from,meta)
    else:
        os.makedirs(save_dir)
        shutil.copyfile(args.config,os.path.join(save_dir,os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')
        
    # 初始化保存训练信息类
    train_history =History(meta['save_dir'])
    
    # 记录初始学习率，详见https://www.bilibili.com/video/BV1WT4y1q7qN
    lr_update_func.before_run(runner)
    time1=datetime.datetime.now()
    #随机采样列表
    # user_groups=mnist_iid(train_dataset,num_users=args.num_users)
    # 训练
    for epoch in range(runner.get('epoch'),runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        runner['epoch'] = epoch + 1
        #确定客户端列表
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  #idxs_users指用户id列表
        print('idxs_users = ',idxs_users)
        local_weights = []
        global_weights = copy.deepcopy(model.state_dict())
        for idx in range(args.num_users):
            model.load_state_dict(global_weights)
            trainloader, testloader = train_loader[idx],val_loader[idx] #根据不同用户id划分数据集并生成trainloader等
            for i in range(args.local_ep):
                if i+1==args.local_ep:
                    jilu=True
                else:
                    jilu=False
                w,loss=trainfed(idx,model,runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), meta,trainloader,jilu)
            # train(model,runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), meta)
            local_weights.append(copy.deepcopy(w))
            validationfed(idx,model,runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta,testloader)
            # local_losses.append(copy.deepcopy(loss))
        train_history.fedafter_epoch(meta)
        # 平均权重
        global_weights = average_weights(local_weights)
        # 更新模型权重
        model.load_state_dict(global_weights)
        validation(model,runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)

    time2=datetime.datetime.now()
    yonshi=time2-time1
    with open('time.txt','a') as f:
        f.write(str(yonshi))
if __name__ == "__main__":
    
    main()
