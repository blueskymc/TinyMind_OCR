#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' main module '

__author__ = 'Ma Cong'

import pandas as pd
import os
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import checkpoint as cp

import data_utils

class base_eval():
    def __init__(self, net, size=(128, 128), channel=1, test='test2\\', train='train\\'):
        self.img_size = size
        self.channel = channel
        self.model = net
        self.testpath = test
        self.cuda_is_ok = False
        net.eval()
        self.filename = os.listdir(test)
        words = os.listdir(train)  # 按时间排序 从早到晚
        self.words = np.array(words)
        self.testnumber = len(self.filename)

    def eval(self, checkpoint_path='checkpoints'):
        checkpoint = cp.load_checkpoint(address=checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])

        test = data_utils.TestSet(self.testpath, self.img_size, self.channel)
        testdatas = test.loadtestdata()
        testdatas.astype(np.float)
        n = 0
        N = 80
        batch_size = 8
        pre = np.array([])
        batch_site = []
        while n < N:
            n += batch_size
            if n < N:
                n1 = n - batch_size
                n2 = n
            else:
                n1 = n2
                n2 = N

            batch_site.append([n1, n2])

        pred_choice = []
        for site in tqdm(batch_site):
            test_batch = testdatas[site[0]:site[1]]
            test_batch = torch.from_numpy(test_batch)
            datas = Variable(test_batch).float()
            datas = datas.view(-1, 1, 128, 128)
            outputs = self.model(datas)
            outputs = outputs.cpu()
            outputs = outputs.data.numpy()
            for out in outputs:
                K = 5
                index = np.argpartition(out, -K)[-K:]
                pred_choice.append(index)
        pre = np.array(pred_choice)
        predicts = []
        for k in range(self.testnumber):
            index = pre[k]
            predict5 = self.words[index]
            predict5 = "".join(predict5)
            predicts.append(predict5)

        dataframe = pd.DataFrame({'filename': self.filename, 'label': predicts})
        dataframe.to_csv("test.csv", index=False, encoding='utf-8')