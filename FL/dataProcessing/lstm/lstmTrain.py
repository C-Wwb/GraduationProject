# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/22 15:06
# @AUTHOR：WUWENBIN
# @FILENAME：lstmTrain.py
# @SOFTNAME：PyCharm

import torch
from torch import nn

from dataPreProcessing.lstmSequence import lstmSequence
from deepLearningCorrelation.buildLSTM import LSTM
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class lstmTrain:
    def train(args, path):
        Dtr, Dte, m, n = lstmSequence.nn_seq(B=args.batch_size)
        input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
        output_size = args.output_size
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        loss_function = nn.MSELoss().to(device)

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # training
        loss = 0
        for i in tqdm(range(args.epochs)):
            cnt = 0
            for (seq, label) in Dtr:
                cnt += 1
                seq = seq.to(device)
                label = label.to(device)
                y_pred = model(seq)
                loss = loss_function(y_pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if cnt % 100 == 0:
                #     print('epoch', i, ':', cnt - 100, '~', cnt, loss.item())
            print('epoch', i, ':', loss.item())

            scheduler.step()

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, path)