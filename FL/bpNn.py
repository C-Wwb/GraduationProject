# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/28 20:43
# @AUTHOR：WUWENBIN
# @FILENAME：bpNn.py
# @SOFTNAME：PyCharm

import numpy as np
from itertools import chain
import torch

from universal.args import args
from deepLearningCorrelation.bpAlgorithm import BP
from dataPreProcessing.loadData import loadData
from dataProcessing.nn.nnTest import nnTest
from dataProcessing.nn.nnTrain import nnTrain

clients = ['Client' + str(i) for i in range(1, 11)]
def main():
    for client in clients:
        nn = BP(args.args_parser(), client)
        nnTrain.train(args.args_parser(), nn)
        nnTest.test(args.args_parser(), nn)
    #print(nn)
    #torch.save(nn.state_dict(), 'nn.pt')

if __name__ == '__main__':
    main()