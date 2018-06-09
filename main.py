#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' main module '

__author__ = 'Ma Cong'

import argparse
import os
import time

from torch.backends import cudnn

import models
import base_train
import base_eval
import vgg16 as vgg
from data_utils import transfer_data

use_gpu = False  # pytorch不支持计算能力小于3的显卡。。。只能靠CPU了

def main():
    '''main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', default='train', required=True, help='train or eval or data')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--epoch', default=10, type=int, help='epoch count')
    parser.add_argument('--checkpoint', default='checkpoint\\', help='模型存储位置')
    parser.add_argument('--dropout', default=0.5, help='drop的概率')
    parser.add_argument('--channel', default=1, type=int, help='图片的通道数')
    parser.add_argument('--imgwidth', default=128, type=int, help='图片预处理后的宽度')
    parser.add_argument('--imgheight', default=128, type=int, help='图片预处理后的高度')
    parser.add_argument('--loadcheckpoints', default=False, help='加载参数模型')
    parser.add_argument('--selftest', default=True, help='在训练样本中抽出一部分作为测试')
    parser.add_argument('--trainpath', default='train\\', help='训练样本文件夹')
    parser.add_argument('--testpath', default='test2\\', help='测试样本文件夹')
    opt = parser.parse_args()
    print(opt)


    if use_gpu:
        cudnn.enabled = False
        # model = model.cuda()

    if opt.state == 'train':
        if opt.channel == 3:
            train = vgg.train_vgg16((opt.imgwidth, opt.imgheight), opt.channel, opt.epoch, opt.batch, opt.selftest)
        elif opt.channel == 1:
            train = base_train.base_train(models.model_channel_one(), (opt.imgwidth, opt.imgheight),
                                          opt.channel, opt.epoch, opt.batch, opt.selftest)
        train.train()

    elif opt.state == 'eval':
        if opt.channel == 3:
            test = vgg.eval_vgg16((opt.imgwidth, opt.imgheight),
                                          opt.channel, opt.testpath, opt.trainpath)
            test.eval()
        elif opt.channel == 1:
            test = base_eval.base_eval(models.model_channel_one(), (opt.imgwidth, opt.imgheight),
                                          opt.channel, opt.testpath, opt.trainpath)
            test.eval()

    elif opt.state == 'data':
        td = transfer_data(opt.trainpath, opt.imgwidth, opt.imgheight, opt.channel is 3)
        td.transData()

    else:
        print('Error state, must choose from train and eval!')


if __name__ == '__main__':
    main()
