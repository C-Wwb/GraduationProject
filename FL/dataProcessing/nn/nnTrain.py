# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/28 21:24
# @AUTHOR：WUWENBIN
# @FILENAME：nnTrain.py
# @SOFTNAME：PyCharm

import numpy as np
from dataPreProcessing.nnSequence import nnSequence
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime

'''df = pd.DataFrame(columns=['time', 'step', 'lossError'])
df.to_csv("T:/assignment/GraduationProject/coding/FL/data/Health/lossError.csv"
          , index=False)#csv'''

class nnTrain:
    def train(args, nn): #模型训练
        print('training...')
        train_x, train_y, test_x, test_y = nnSequence.nn_sequence(nn.file_name, args.B) #test_y = 全局模型
        nn.len = len(train_x)
        batch_size = args.B #batch_size每批数据量大小；args动态参数，任意数量个参数
        epochs = args.E #epochs 训练过程中数据将被轮多少次
        batch = int(len(train_x) / batch_size) #深度学习的优化算法
        for epoch in range(epochs):
            for i in range(batch):
                start = i * batch_size
                end = start + batch_size
                nn.forward_prop(train_x[start:end], train_y[start:end]) #正向传播计算
                nn.backward_prop(train_y[start:end]) #反向传播计算
            print('epoch:', epoch, ' error:', np.mean(nn.loss)) #第epoch轮的错误损失平均值

        return nn

''' ml = np.mean(nn.loss)
            time = "%s" % datetime.now()
            step = "Step[%d]" % epoch
            lossError = "%f" % ml
            list = [time, epoch, lossError]
            data = pd.DataFrame([list])
            data.to_csv('T:/assignment/GraduationProject/coding/FL/data/Health/lossError.csv'
                        , mode='a', header=False, index=False)  # csv'''