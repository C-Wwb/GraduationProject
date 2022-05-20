# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/30 15:51
# @AUTHOR：WUWENBIN
# @FILENAME：popa.py
# @SOFTNAME：PyCharm

from deepLearningCorrelation.buildNeuralNetwork import ANN

import torch

device = torch.device('cpu')
net = torch.load('model.pt', map_location = device)
torch.save(net, 'model-cpu.pt')