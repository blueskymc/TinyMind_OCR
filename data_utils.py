#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'picture preprocess and generate middle file'

import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from tqdm import tqdm

class transfer_data():
    def __init__(self, p = 'train\\', w=128, h=128, channel3=False):
        self.data_path = p
        self.words = os.listdir(p)
        self.img_size = (w, h)
        self.channel_is_3 = channel3

    def loadOneWord(self, order):
        path = self.data_path + self.words[order] + '\\'
        files = os.listdir(path)
        datas = []
        for file in files:
            file = path + file
            img = Image.open(file)
            if self.channel_is_3:
                img_pic_L = img.convert("L")
                img = img_pic_L.convert("RGB")
            img = np.asarray(img)
            img = cv2.resize(img, self.img_size)
            datas.append(img)
        datas = np.array(datas)
        labels = np.zeros([len(datas), len(self.words)], dtype=np.uint8)
        labels[:, order] = 1
        return datas, labels

    def transData(self):
        num = len(self.words)
        datas = np.array([], dtype=np.uint8)
        # datas.shape = -1, self.img_size[0], self.img_size[1], 3 if self.channel_is_3 else 1
        if self.channel_is_3:
            datas.shape = -1, self.img_size[0], self.img_size[1], 3
        else:
            datas.shape = -1, self.img_size[0], self.img_size[1]
        labels = np.array([], dtype=np.uint8)
        labels.shape = -1, 100
        for k in tqdm(range(num)):
            data, label = self.loadOneWord(k)

            datas = np.append(datas, data, axis=0)
            labels = np.append(labels, label, axis=0)

        np.save('data\\data.npy', datas)
        np.save('data\\label.npy', labels)

class TrainSetForCompetition(data.Dataset):
    def __init__(self):
        datas = np.load('data\\data.npy')
        labels = np.load('data\\label.npy')
        index = np.arange(0, len(datas), 1, dtype=np.int)
        np.random.seed(123)
        np.random.shuffle(index)
        self.data = datas[index]
        self.label = labels[index]
        np.random.seed()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), \
               torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

class TrainSetForSelftest(data.Dataset):
    def __init__(self, eval):
        datas = np.load('data\\data.npy')
        labels = np.load('data\\label.npy')
        index = np.arange(0, len(datas), 1, dtype=np.int)
        np.random.seed(123)
        np.random.shuffle(index)
        if eval:
            index = index[:int(len(datas) * 0.1)]
        else:
            index = index[int(len(datas) * 0.1):]
        self.data = datas[index]
        self.label = labels[index]
        np.random.seed()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), \
               torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)

class TestSet(data.Dataset):
    def __init__(self, path, img_size=(128,128), channel3=False):
        self.path = path
        self.img_size = img_size
        self.channel_is_3 = channel3

    def loadtestdata(self):
        files = os.listdir(self.path)
        datas = []
        for file in tqdm(files):
            file = self.path + file
            img = Image.open(file)
            if self.channel_is_3:
                img_pic_L = img.convert("L")
                img = img_pic_L.convert("RGB")
            img = np.asarray(img)
            img = cv2.resize(img, self.img_size)
            datas.append(img)
        datas = np.array(datas)
        return datas