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

from options import args_parser
from update import LocalUpdate, test_inference
from models2 import CNN_1,CNN_2,MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from get_data import get_data2
from pytorchtools import EarlyStopping

if __name__ == '__main__':
    start_time = time.time()
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []


    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device='cpu'
    # if args.dataset=='master':
    t=0
    #train_dataset, test_dataset, _ = get_dataset(args)
    x_train, y_train,x_test, y_test=get_data2(t)
    #转换numpy为tensor
    if t==3:
        x_train=x_train.reshape(-1,1,16,54)
        x_test=x_test.reshape(-1,1,16,54)
    else:
        x_train=x_train.reshape(-1,1,16,72)
        x_test=x_test.reshape(-1,1,16,72)
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
    testloader = DataLoader(test_dataset, batch_size=6294, shuffle=True)
    user_groups=mnist_iid(train_dataset,100)
    early_stopping = EarlyStopping(patience=0.001, verbose=True)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        global_model = CNN_2()
    elif args.model == 'cnn_1':
        global_model = CNN_1()
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
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    criterion = torch.nn.CrossEntropyLoss().to('cpu')
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        valid_losses=[]
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
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
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
                ######################    
        # validate the model #
        ######################
        
        for data, target in testloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = global_model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, global_model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    torch.save(global_model,'./global_model')
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    #file_name = 'E:/save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #           args.local_ep, args.local_bs)

    #with open(file_name, 'wb') as f:
    #    pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Qt5Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(1,len(train_loss)+1), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    # plt.savefig('../fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    plt.show()
    # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    # plt.show()
with torch.no_grad():
    correct = 0
    total = 0
    a=[]
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = global_model(images)
        _, predicted = torch.max(outputs.data, 1)#预测结果
        total += labels.size(0)#正确结果
        correct += (predicted == labels).sum().item() #正确结果总数
 
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


    # from sklearn.metrics import confusion_matrix,classification_report
    # from sklearn.metrics import recall_score
    # import matplotlib.pyplot as plt

    # guess = predicted.data.numpy()
    # fact = labels.data.numpy()
    # print(classification_report(fact,guess,digits=5))
    # classes = list(set(fact))
    # classes.sort()
    # confusion = confusion_matrix(guess, fact)
    # plt.imshow(confusion, cmap=plt.cm.Blues)
    # indices = range(len(confusion))
    # A=['drtv','hbo','http','https','netflix','twitch','youtube']
    # plt.xticks(indices, A)
    # plt.yticks(indices, A)
    # plt.colorbar()
    # plt.xlabel('predict')
    # plt.ylabel('fact')
   
    # for first_index in range(len(confusion)):
    #     for second_index in range(len(confusion[first_index])):
    #         plt.text(first_index, second_index, confusion[first_index][second_index])
    # plt.show()