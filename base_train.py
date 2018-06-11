#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'base class for train'

__author__ = 'Ma Cong'

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from datetime import datetime

import checkpoint as cp
import data_utils

class base_train():
    def __init__(self, net, size=(128,128), channel=1, epoch_count=10, batch_size=256, selftest=True):
        self.img_size = size
        self.channel = channel
        self.n_epoch = epoch_count
        self.model = net
        self.selftest = selftest
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
        if selftest:
            trainset = data_utils.TrainSetForSelftest(eval=False)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            evalset = data_utils.TrainSetForSelftest(eval=True)
            self.evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)
        else:
            trainset = data_utils.TrainSetForCompetition()
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    def train(self, checkpoint_path='checkpoints'):
        # 是否装载模型参数
        load = False

        if load:
            checkpoint = cp.load_checkpoint(address=checkpoint_path)
            checkpoint.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.n_epoch):
            self.train_one_epoch(epoch)

            # 保存参数
            checkpoint = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
            cp.save_checkpoint(checkpoint, address=checkpoint_path)

            if self.selftest:
                self.eval(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()  # 网络处于训练模式，会导致dropout启用
        correct = 0
        sum = 0
        T = 0

        print(now())
        print('Begin training...')
        for batch_index, (datas, labels) in enumerate(self.trainloader, 0):
            labels = labels.max(1)  # 对行取最大值，得到二维元组，0列为值，1列为索引，注意此处tensor与numpy的max是不一样的
            labels = labels[1]  # 取最大值所在列的索引
            datas = Variable(datas).float()  # 创建变量

            # torch.nn 只接受小批量的数据， 维度是[batch，channel，height，width]
            # 此处-1是根据其他维数来确定的，例如原来size为[512, 128, 128]
            # 经过-1转换为[512, 1, 128, 128]，用来对应torch.nn的输入变量
            datas = datas.view(-1, self.channel, self.img_size[0], self.img_size[1])
            labels = Variable(labels).long()
            # if cuda_is_ok:
            #     datas = datas.cuda()
            #     labels = labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            T += 1
            pred_choice = outputs.data.max(1)[1]
            correct += pred_choice.eq(labels.data).cpu().sum()
            sum += len(labels)

            print('batch_index: [%d/%d]' % (batch_index, len(self.trainloader)),
                  'Train epoch: [%d]' % (epoch),
                  'correct/sum:%d/%d, %.4f' % (correct, sum, correct / sum))
            print(now())

    def eval(self, epoch):
        self.model.eval()  # 弯网络处于测试模式，dropout停用，BN放射变换停止
        correct = 0
        sum = 0
        for batch_index, (datas, labels) in enumerate(self.evalloader, 0):
            labels = labels.max(1)[1]
            datas = Variable(datas).float()
            datas = datas.view(-1, self.channel, self.img_size[0], self.img_size[1])
            labels = Variable(labels).long()
            outputs = self.model(datas)

            pred_choice = outputs.data.max(1)[1]
            correct += pred_choice.eq(labels.data).cpu().sum()
            sum += len(labels)
            print('batch_index: [%d/%d]' % (batch_index, len(self.evalloader)),
                  'Eval epoch: [%d]' % (epoch),
                  'correct/sum:%d/%d, %.4f' % (correct, sum, correct / sum))

def now():
    return datetime.now().strftime('%c')