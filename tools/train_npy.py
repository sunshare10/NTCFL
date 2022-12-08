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
import numpy as np
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

def main():
    # 读取配置文件获取关键字段
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    print_info(model_cfg)

    # 初始化
    meta = dict()
    NUM_HEADERS=16
    PCK_LEN=1500
    PNG_chan=160  #灰度图尺寸
    niming_flag=1  #0表示不匿名，1表示匿名不带方向，2表示匿名且标识方向,3表示用255进行匿名，不带方向
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t=1   #t表示实验4
    dirname = str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('logs',model_cfg.get('backbone').get('type'),dirname)
    meta['save_dir'] = save_dir

    
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
    
    x_train, y_train,x_test, y_test,train_dataset,test_dataset,train_loader,val_loader ={},{},{},{},{},{},{},{}
    x_std,y={},{}
    for idx in range(args.num_users):
        data = np.load('./datasets1/client'+'_'+str(idx)+'_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+'.npz')
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
            x_train_all=x_train[idx]
            y_train_all=y_train[idx]
        else:
            x_test_all= np.concatenate((x_test_all, x_test[idx]), axis=0)
            y_test_all= np.concatenate((y_test_all, y_test[idx]), axis=0)
            x_train_all= np.concatenate((x_train_all, x_train[idx]), axis=0)
            y_train_all= np.concatenate((y_train_all, y_train[idx]), axis=0)

    x_train=torch.from_numpy(x_train_all)
    y_train=torch.from_numpy(y_train_all)
    x_test=torch.from_numpy(x_test_all)
    y_test=torch.from_numpy(y_test_all)
    x_train=x_train.type(torch.FloatTensor)
    y_train=y_train.type(torch.LongTensor)
    x_test=x_test.type(torch.FloatTensor)
    y_test=y_test.type(torch.LongTensor)
    #封装数据
    # print(x_train[0])
    train_dataset=TensorDataset(x_train,y_train)
    test_dataset=TensorDataset(x_test,y_test)
    train_loader = DataLoader(train_dataset,  shuffle=True, batch_size=96, num_workers=data_cfg.get('num_workers'), pin_memory=False,
    drop_last=True)
    val_loader = DataLoader(test_dataset,  shuffle=True, batch_size=96, num_workers=data_cfg.get('num_workers'), pin_memory=False,
    drop_last=True)
    
    
    # 将关键字段存储，方便训练时同步调用&更新
    runner = dict(
        optimizer         = optimizer,
        train_loader      = train_loader,
        val_loader        = val_loader,
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
    # 训练
    for epoch in range(runner.get('epoch'),runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        runner['epoch'] = epoch + 1
        train(model,runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), meta)
        validation(model,runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)
        train_history.after_epoch(meta)
    time2=datetime.datetime.now()
    yonshi=time2-time1
    # with open('time.txt','w') as f:
    #     f.write(str(yonshi))
if __name__ == "__main__":
    
    main()
