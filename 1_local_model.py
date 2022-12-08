#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# from utils import get_dataset
from options import args_parser
from update import test_inference
from models2 import CNN_1,CNN_2,CNN_3, MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from get_data import get_data2
from torch.utils.data import Dataset,TensorDataset
# from imblearn.over_sampling import RandomOverSampler
# from petastorm import make_reader
from utils import plot_ROC
from model import ResnetRS

if __name__ == '__main__':
    args = args_parser()
    NUM_HEADERS=16
    PCK_LEN=1500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    t=10
    
    x_train, y_train,x_test, y_test=get_data2(num_headers = NUM_HEADERS,pck_len=PCK_LEN,t=t)
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
    elif PCK_LEN==60:
        x_train=x_train.reshape(-1,1,60,60)
        x_test=x_test.reshape(-1,1,60,60)
    elif NUM_HEADERS==16 and PCK_LEN==1500:
        x_train=x_train.reshape(-1,1,160,160)
        x_test=x_test.reshape(-1,1,160,160)
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
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # else:
    #     data_path = 'train_test_data/traffic_classification/train.parquet/part-00000-bfc29d6e-9c8e-4ce1-abef-cc706c10429f-c000.snappy.parquet'
    #     da4=Path(data_path).absolute().as_uri()
    #     reader = make_reader(da4)
        # reader = make_reader('file:///application_classification/train.parquet')
        # trainloader=
    # BUILD MODEL
    if args.model == 'cnn' and PCK_LEN==72:
        # Convolutional neural netork
        global_model = CNN_2()
        # global_model = ResnetRS.create_pretrained('resnetrs152', in_ch=1, num_classes=20,
                        #    drop_rate=0.01)
    elif args.model == 'cnn' and PCK_LEN==1500:
        global_model = CNN_3()
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)
    a=np.array(epoch_loss)
    np.save('local.npy',a)
    # Plot loss
    plt.figure()
    plt.plot(range(1,len(epoch_loss)+1), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    #plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,args.epochs))
    plt.show()
    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
    # from sklearn.metrics import classification_report #失败
    # cr=classification_report(global_model(x_test).data.numpy(),y_test.data.numpy())

    

with torch.no_grad():
    correct = 0
    total = 0
    a=[]
    # predictlist=[]
    # lablelist=[]
    global_model.to(device)
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = global_model(images)
        _, predicted = torch.max(outputs.data, 1)#预测结果
        if total==0:
            predictlist=predicted
            lablelist=labels
        else:
            predictlist=torch.cat((predictlist,predicted))
            lablelist=torch.cat((lablelist,labels))
        total += labels.size(0)#正确结果
        correct += (predicted == labels).sum().item() #正确结果总数
        
 
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    from sklearn.metrics import confusion_matrix,classification_report
    import matplotlib.pyplot as plt
    
    
    guess = predictlist.data.cpu().numpy()
    fact = lablelist.data.cpu().numpy()
    print(classification_report(fact,guess,digits=5))
    # 试图绘制ROC曲线失败
    # num_classes=7
    # labels=['drtv','hbo','http','https','netflix','twitch','youtube']
    # plot_ROC(fact, guess, num_classes, labels, micro=False, macro=False)
    classes = list(set(fact))
    classes.sort()
    confusion = confusion_matrix(guess, fact)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('guess')
    plt.ylabel('fact')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    plt.show()