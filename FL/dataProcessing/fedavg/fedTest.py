# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/27 19:56
# @AUTHOR：WUWENBIN
# @FILENAME：fedTest.py
# @SOFTNAME：PyCharm

import sys
import torch
from itertools import chain
import numpy as np
from datetime import datetime
import pandas as pd

sys.path.append('../')
from sklearn.metrics import mean_absolute_error, mean_squared_error #mae：平均绝对误差 rmse：均方根误差
from dataPreProcessing.fedavgSequence import structuralSequence


clients = ['Client' + str(i) for i in range(1, 11)]
#Client1-10
'''df = pd.DataFrame(columns=['time', 'mae', 'rmse'])
df.to_csv("T:/assignment/GraduationProject/coding/FL/data/Health/maeRmse.csv"
          , index=False)#csv'''

class fedTest:
    def test(args, model): #测试
        #global MAX, MIN
        model.eval()
        Dtr, Dte = structuralSequence.nn_sequence(model.name, args.B)
        pred = []
        y = []
        device = args.device
        for (seq, target) in Dte:
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
                y.extend(list(chain.from_iterable(target.data.tolist())))
        #
        pred = np.array(pred)
        y = np.array(y)

        '''m = mean_absolute_error(y, pred)
        r = np.sqrt(mean_squared_error(y, pred))
        time = "%s" % datetime.now()
        mae = "%f" % m
        rmse = "%f" % r
        List = [time, mae, rmse]
        data = pd.DataFrame([List])
        data.to_csv('T:/assignment/GraduationProject/coding/FL/data/Health/maeRmse.csv'
                    , mode='a', header=False, index=False)  #mae rmse'''

        print('mae:', mean_absolute_error(y, pred), 'rmse:',
              np.sqrt(mean_squared_error(y, pred)))  # mae：平均绝对误差 rmse：均方根误差
        print('mape:', np.mean(np.abs((pred-y) / pred)))





