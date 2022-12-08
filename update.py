#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from unittest import TestLoader
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from core.evaluations import evaluate
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


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        #self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #    dataset, list(idxs))
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs)) #根据不同用户id划分数据集并生成trainloader等
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        #return trainloader, validloader, testloader
        return trainloader, testloader

    def update_weights(self, model, global_round,optimizer_cfg,runner,lr_update_func):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # elif self.args.optimizer == 'AdamW':
            # optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(),**optimizer_cfg)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                with torch.no_grad():
                    images, labels = images.to(self.device), labels.to(self.device)
                runner.get('optimizer').zero_grad()
                #Trainloader确定训练iter次数，影响学习率更新，需进行更新
                lr_update_func.before_train_iter(runner)
                # log_probs = model(images)
                losses = model(images,targets=labels,return_loss=True)
                # loss = self.criterion(log_probs, labels)
                losses.get('loss').backward()
                runner.get('optimizer').step()
                # loss.backward()
                # optimizer.step()
                runner['iter'] += 1
                self.logger.add_scalar('loss', losses.get('loss').item())
                batch_loss.append(losses.get('loss').item())
            epoch_loss_a=sum(batch_loss)/len(batch_loss)
            print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), epoch_loss_a))
            epoch_loss.append(epoch_loss_a)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model,cfg):
        """ Returns the inference accuracy and loss.
        """
        loss=0.0
        preds,targets2 = [],[]
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for iter, batch in enumerate(self.testloader):
                images, target = batch
                images, target = images.to(self.device),target.to(self.device)
                outputs = model(images,return_loss=False)
                preds.append(outputs)
                targets2.append(target) 
                batch_losses = model(images,targets=target,return_loss=True)
                loss += batch_losses.get('loss').item()
            loss=loss/len(self.testloader)
        eval_results = evaluate(torch.cat(preds),torch.cat(targets2),cfg.get('metrics'),cfg.get('metric_options'))
        accuracy=eval_results.get('accuracy_top-1',0.0)
        
        # model.eval()
        # loss, total, correct = 0.0, 0.0, 0.0

        # for batch_idx, (images, labels) in enumerate(self.testloader):
        #     images, labels = images.to(self.device), labels.to(self.device)

        #     # Inference
        #     batch_loss = model(images,targets=labels,return_loss=True)
        #     outputs = model(images,return_loss=False)
        #     # batch_loss = self.criterion(outputs, labels)
        #     loss += batch_loss.item()

        #     # Prediction
        #     _, pred_labels = torch.max(outputs, 1)
        #     pred_labels = pred_labels.view(-1)
        #     correct += torch.sum(torch.eq(pred_labels, labels)).item()
        #     total += len(labels)

        # accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset,cfg):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    device = 'cuda:0' 
    # criterion = nn.NLLLoss().to(device)
    model.to(device)
    testloader = DataLoader(test_dataset, batch_size=32,
                            shuffle=False)
    preds,targets2 = [],[]

    model.to(device)
    with torch.no_grad():
        for iter, batch in enumerate(testloader):
            images, target = batch
            images, target = images.to(device),target.to(device)
            outputs = model(images,return_loss=False)
            preds.append(outputs)
            targets2.append(target) 
            batch_loss = model(images,targets=target,return_loss=True)
            loss += batch_loss.get('loss').item()
        loss=loss/(iter+1)
    eval_results = evaluate(torch.cat(preds),torch.cat(targets2),cfg.get('metrics'),cfg.get('metric_options'))
    accuracy=eval_results.get('accuracy_top-1',0.0)
    return accuracy, loss
