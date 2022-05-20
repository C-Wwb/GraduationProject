# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/18 15:43
# @AUTHOR：WUWENBIN
# @FILENAME：tensorboardDraw.py
# @SOFTNAME：PyCharm

import sys
import torch
import pandas as pd
import csv

sys.path.append('../')
from torch import nn #全局模型

from dataPreProcessing.fedavgSequence import structuralSequence
from universal.args import args
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

folder = 'T:/assignment/GraduationProject/coding/FL/data/Health/fig'
writer = SummaryWriter(log_dir=folder, flush_secs=30) #tensorboard

with open('../data/Health/client1.csv', 'r') as c:
    #print(c.read())
    r = csv.reader(c)
    index = 0
    for i in r:
        if (index != 0):
            loss = i[2]
            epoch = i[1]
            writer.add_scalar('loss', loss, epoch)  # tensorboard
            #print(i[1], ' ', i[2])
        index += 1