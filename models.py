#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'create models'

__author__ = 'Ma Cong'

import torch
import torch.nn as nn
from torchvision import models

cuda_is_ok = False

class model_channel_one(nn.Module):
    def __init__(self):
        super(model_channel_one, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(8*8*128, 1024),
            nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 100),
        )

    def forward(self, input):
        x = self.Conv(input)
        x = x.view(-1, 8*8*128)
        x = self.Classes(x)
        return x

