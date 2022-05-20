# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/27 19:55
# @AUTHOR：WUWENBIN
# @FILENAME：fedTrain.py
# @SOFTNAME：PyCharm

import sys
import torch
import pandas as pd

sys.path.append('../')
from torch import nn #全局模型

from dataPreProcessing.fedavgSequence import structuralSequence
from universal.args import args
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

clients = ['Client' + str(i) for i in range(1, 11)]
#Client1-10
#Loss_list = []
folder = 'T:/assignment/GraduationProject/coding/FL/data/Health/fig'
writer = SummaryWriter(log_dir=folder, flush_secs=30) #tensorboard

'''df = pd.DataFrame(columns=['time', 'step', 'trainLoss'])
df.to_csv("T:/assignment/GraduationProject/coding/FL/data/Health/lossData.csv"
          , index=False)#csv'''

class fedTrain:
    def train(args, model, global_round): #训练
        model.train()
        Dtr, Dte = structuralSequence.nn_sequence(model.name, args.B)
        model.len = len(Dtr)
        device = args.device
        loss_function = nn.MSELoss().to(device)
        loss = 0
        if args.weight_decay != 0:
            lr = args.lr * pow(args.weight_decay, global_round)
        else:
            lr = args.lr
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.9, weight_decay=args.weight_decay)

        for epoch in range(args.E): #E：两次联邦之间的本地训练次数
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
                #print(cnt)
            #Loss_list.append(loss.item())
            print('epoch', epoch, ':', loss.item())


            writer.add_scalar('loss', loss.item(), epoch)#tensorboard

            '''time = "%s" % datetime.now()
            step = "Step[%d]" % epoch
            train_loss = "%f" % loss.item()
            list = [time, step, train_loss]
            data = pd.DataFrame([list])
            data.to_csv('T:/assignment/GraduationProject/coding/FL/data/Health/lossData.csv'
                        , mode='a', header=False, index=False)  # csv
        #print(Loss_list)'''

        return model

'''
def main():
    drawLoss.draw_loss(Loss_list, 20)

if __name__ == '__main__':
    main()

'''
