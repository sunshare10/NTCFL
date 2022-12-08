#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from sampling import mnist_iid
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,TensorDataset
from torch.utils.data import DataLoader
from models.build import BuildNet
from utils.inference import init_model
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from core.optimizers import *
import torch.optim as optim
from options import args_parser
from update import LocalUpdate, test_inference
from models2 import CNN_1,CNN_2, MLP,CNN_160
from utils3 import average_weights, exp_details
from get_data import get_data2
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    
    args = args_parser()
    exp_details(args)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device='cpu'


    NUM_HEADERS=16
    PCK_LEN=1500
    PNG_chan=160  #灰度图尺寸
    niming_flag=1  #0表示不匿名，1表示匿名不带方向，2表示匿名且标识方向,3表示用255进行匿名，不带方向
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t=1
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)
    x_train, y_train,x_test, y_test=get_data2(num_headers = NUM_HEADERS,pck_len=PCK_LEN,pngchang=PNG_chan,t=t,niming_flag=niming_flag)
    # ros = RandomOverSampler(random_state=0)
    # x_train, y_train = ros.fit_resample(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    #转换numpy为tensor
    if NUM_HEADERS==16 and PCK_LEN==54:
        x_train=x_train.reshape(-1,1,60,60)
        x_test=x_test.reshape(-1,1,60,60)
    elif NUM_HEADERS==16 and PCK_LEN==72:
        x_train=x_train.reshape(-1,1,60,60)
        x_test=x_test.reshape(-1,1,60,60)
    elif NUM_HEADERS==8 and PCK_LEN==1500:
        x_train=x_train.reshape(-1,1,112,112)
        x_test=x_test.reshape(-1,1,112,112)
    elif NUM_HEADERS==16 and PCK_LEN==1500:
        x_train=x_train.reshape(-1,1,160,160)
        x_test=x_test.reshape(-1,1,160,160)
    elif NUM_HEADERS==24 and PCK_LEN==1500:
        x_train=x_train.reshape(-1,1,192,192)
        x_test=x_test.reshape(-1,1,192,192)
    elif NUM_HEADERS==32 and PCK_LEN==1500:
        x_train=x_train.reshape(-1,1,224,224)
        x_test=x_test.reshape(-1,1,224,224)
    elif NUM_HEADERS==40 and PCK_LEN==1500:
        x_train=x_train.reshape(-1,1,280,280)
        x_test=x_test.reshape(-1,1,280,280)
    elif NUM_HEADERS==48 and PCK_LEN==1500:
        x_train=x_train.reshape(-1,1,288,288)
        x_test=x_test.reshape(-1,1,288,288)

    x_train=torch.from_numpy(x_train)
    y_train=torch.from_numpy(y_train)
    x_test=torch.from_numpy(x_test)
    y_test=torch.from_numpy(y_test)
    x_train=x_train.type(torch.FloatTensor)
    y_train=y_train.type(torch.LongTensor)
    x_test=x_test.type(torch.FloatTensor)
    y_test=y_test.type(torch.LongTensor)
    #封装数据
    # print(x_train[0])
    train_dataset=TensorDataset(x_train,y_train)
    test_dataset=TensorDataset(x_test,y_test)
    trainloader = DataLoader(train_dataset,  shuffle=True, batch_size=32, num_workers=data_cfg.get('num_workers'), pin_memory=True,
    drop_last=True)
    testloader = DataLoader(test_dataset,  shuffle=True, batch_size=32, num_workers=32, pin_memory=True,
    drop_last=True)
    # testloader = DataLoader(test_dataset, batch_size=6294, shuffle=True)
    user_groups=mnist_iid(train_dataset,num_users=args.num_users)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        global_model = CNN_2()
    elif args.model == 'cnn_1':
        global_model = CNN_1()
    elif args.model == 'cnn_160':
        global_model = CNN_160()
    elif args.model=='Transformer':
        # global_model=transformer()
        global_model = BuildNet(model_cfg)
        global_model.init_weights()
        print(global_model)
        # global_model = init_model(global_model, data_cfg, device=device, mode='train')
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        
        for x in img_size:
            len_in *= x
            # print('x:',x,'len_in:',len_in)
        global_model = MLP(dim_in=len_in, dim_hidden=64,dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=global_model.parameters(),**optimizer_cfg)
    # copy weights
    global_weights = global_model.state_dict()
    runner = dict(
        optimizer         = optimizer,
        train_loader      = trainloader,
        val_loader        = testloader,
        iter              = 0,
        epoch             = 0,
        max_epochs       = args.epochs,
        max_iters         = data_cfg.get('train').get('epoches')*len(trainloader),
        best_train_loss   = float('INF'),
        best_val_acc     = float(0),
        best_train_weight = '',
        best_val_weight   = '',
        last_weight       = ''
    )
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    lr_update_func.before_run(runner)
    for epoch in tqdm(range(args.epochs)):
        runner['epoch'] = epoch + 1
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        lr_update_func.before_train_epoch(runner)   #确定每轮epoch训练前的更新率
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  #idxs_users指用户id列表
        print('idxs_users = ',idxs_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), 
                                                 global_round=epoch,optimizer_cfg=optimizer_cfg,runner=runner,lr_update_func=lr_update_func)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model,cfg=data_cfg.get('test'))
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset,cfg=data_cfg.get('test'))
    torch.save(global_model,'./global_model')
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))



    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
