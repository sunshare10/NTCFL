#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import torch.nn.functional as F
from pathlib import Path

# from petastorm import make_reader
# from petastorm.pytorch import DataLoader
# from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
#        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        return self.softmax(x)
class CNN_3(nn.Module):
    def __init__(self):#, args
        super(CNN_3, self).__init__()
        self.conv1 = nn.Sequential(#1*16*54   1*160*160
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,stride=2,padding=1),#16*16*54   16*16*72
            nn.ReLU(),
            # nn.MaxPool2d(2) #16-80-80
        )
        self.conv2 = nn.Sequential(#16*8*27    8*8*72
            # nn.Conv2d(16, 32, kernel_size=(3,4),stride=1,padding=0),#32*6*24 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=2,padding=1),#32*80*80
            nn.ReLU(),
            # nn.MaxPool2d(2)#32*40*40
        )
        self.conv3 = nn.Sequential(#16*8*27    8*8*72
            # nn.Conv2d(16, 32, kernel_size=(3,4),stride=1,padding=0),#32*6*24 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2,padding=1),#32*80*80
            nn.ReLU(),
            # nn.MaxPool2d(2)#64*20*20
        )
        self.fc1 = nn.Linear(25600,1000)
        self.fc2 = nn.Linear(1000, 100)

    def forward(self, x):
        # x=x.reshape(64,1,16,54)/255
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        out=self.fc2(x)
        return F.log_softmax(out, dim=1)
 
class CNN_2(nn.Module):
    def __init__(self):#, args
        super(CNN_2, self).__init__()
        self.conv1 = nn.Sequential(#1*16*54   1*16*72 1-60-60
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,stride=1,padding=1),#16*16*54   16*16*72  16-72-72
            nn.ReLU(),
            nn.MaxPool2d(2) #16-36-36 16-30-30
        )
        self.conv2 = nn.Sequential(#16*8*27    8*8*72
            # nn.Conv2d(16, 32, kernel_size=(3,4),stride=1,padding=0),#32*6*24 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1),#32*16*72 32 36 36
            nn.ReLU(),
            nn.MaxPool2d(2)#32*8*36 32 18 18 32-15-15
        )
        self.conv3 = nn.Sequential(#16*8*27    8*8*72  
            # nn.Conv2d(16, 32, kernel_size=(3,4),stride=1,padding=0),#32*6*24 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,padding=1),#32*16*72 32 36 36
            nn.ReLU(),
            nn.MaxPool2d(2)#32*8*36 32 18 18 64 7 7
        )
        self.fc1 = nn.Linear(3136,1000)
        self.fc2 = nn.Linear(1000, 20)

    def forward(self, x):
        # x=x.reshape(64,1,16,54)/255
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        out=self.fc2(x)
        return F.log_softmax(out, dim=1)

class CNN_160(nn.Module):
    def __init__(self):#, args
        super(CNN_160, self).__init__()
        self.conv1 = nn.Sequential(#160×160×1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,stride=4,padding=1),#40×40×32
            nn.ReLU(),
            nn.MaxPool2d(2) #20×20×32
        )
        self.conv2 = nn.Sequential(#16*8*27
            # nn.Conv2d(16, 32, kernel_size=(3,4),stride=1,padding=0),#32*6*24 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1),#10×10×128
            nn.ReLU(),
            nn.MaxPool2d(2)#5×5×128
        )
        self.fc1 = nn.Linear(3200, 50)
        self.fc2 = nn.Linear(50, 12)

    def forward(self, x):
        # x=x.reshape(64,1,16,54)/255
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        out=self.fc2(x)
        return F.log_softmax(out, dim=1)  
    
class CNN_1(nn.Module):
    def __init__(self):#, args
        super(CNN_1, self).__init__()
        self.conv1 = nn.Sequential(#1*16*54
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,stride=1,padding=1),#16*16*54
            nn.ReLU(),
            # nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(#16*8*27
            # nn.Conv2d(16, 32, kernel_size=(3,4),stride=1,padding=0),#32*6*24 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1),#32*16*54
            nn.ReLU(),
            nn.MaxPool2d(2)#32*8*27
        )
        self.fc1 = nn.Linear(6912, 50)
        self.fc2 = nn.Linear(50, 16)

    def forward(self, x):
        # x=x.reshape(64,1,16,54)/255
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        out=self.fc2(x)
        return F.log_softmax(out, dim=1)   
    
# cnn=CNN_2()
# print(cnn)
'''
class CNN_3(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # config
        self.hparams = hparams
        self.data_path = self.hparams.data_path

        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hparams.c1_output_dim,
                kernel_size=self.hparams.c1_kernel_size,
                stride=self.hparams.c1_stride
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hparams.c1_output_dim,
                out_channels=self.hparams.c2_output_dim,
                kernel_size=self.hparams.c2_kernel_size,
                stride=self.hparams.c2_stride
            ),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=2
        )

        # flatten, calculate the output size of max pool
        # use a dummy input to calculate
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # followed by 5 dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=max_pool_out,
                out_features=200
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=200,
                out_features=100
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(
                in_features=100,
                out_features=50
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(
            in_features=50,
            out_features=self.hparams.output_dim
        )

    def forward(self, x):
        # make sure the input is in [batch_size, channel, signal_length]
        # where channel is 1
        # signal_length is 1500 by default
        batch_size = x.shape[0]

        # 2 conv 1 max
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = x.reshape(batch_size, -1)

        # 3 fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # output
        x = self.out(x)

        return x

    def train_dataloader(self):
        reader = make_reader(Path(self.data_path).absolute().as_uri(), reader_pool_type='process', workers_count=12,
                             pyarrow_serialize=True, shuffle_row_groups=True, shuffle_row_drop_partitions=2,
                             num_epochs=self.hparams.epoch)
        dataloader = DataLoader(reader, batch_size=16, shuffling_queue_capacity=4096)

        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y = batch['label'].long()
        y_hat = self(x)

        loss = {'loss': F.cross_entropy(y_hat, y)}

        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss, step=batch_idx)
        return loss
'''
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(200, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
