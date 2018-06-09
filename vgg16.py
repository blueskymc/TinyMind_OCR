#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'train by vgg16'

__author__ = 'Ma Cong'

import torch.optim as optim
from torchvision import models
import torch
import torch.nn as nn

from base_train import base_train
from base_eval import base_eval

model_vgg16 = models.vgg16(pretrained=True)
for param in model_vgg16.parameters():
    param.requires_grad = False

model_vgg16.classifier = torch.nn.Sequential(
    torch.nn.Linear(512 * 2 * 2, 512),
    nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(512, 512),
    nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(512, 100),
)

class train_vgg16(base_train):
    def __init__(self, size=(128,128), channel=1, epoch_count=10, batch_size=256, selftest=True):
        super(train_vgg16, self).__init__(model_vgg16, size, channel, epoch_count, batch_size, selftest)
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)

class eval_vgg16(base_eval):
    def __init__(self, size=(128,128), channel=1, test='test2\\', train='train\\'):
        super(eval_vgg16, self).__init__(model_vgg16, size, channel, test, train)