# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/28 21:22
# @AUTHOR：WUWENBIN
# @FILENAME：nnTest.py
# @SOFTNAME：PyCharm

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataPreProcessing.nnSequence import nnSequence
from itertools import chain
import pandas as pd
from datetime import datetime

'''df = pd.DataFrame(columns=['time', 'mae', 'rmse'])
df.to_csv("T:/assignment/GraduationProject/coding/FL/data/Health/mr.csv"
          , index=False)#csv'''

class nnTest:
    def test(args, nn):  # 模型测试
        train_x, train_y, test_x, test_y = nnSequence.nn_sequence(nn.file_name, args.B)  # test_y = 全局模型
        pred = []
        batch = int(len(test_y) / args.B)  # 深度学习优化算法
        for i in range(batch):
            start = i * args.B
            end = start + args.B
            res = nn.forward_prop(test_x[start:end], test_y[start:end])  # 正向传播计算
            res = res.tolist()
            res = list(chain.from_iterable(res))  # 将多个迭代器进行高效链接，将不同res拼起来并转换成列表
            # print('res=', res)
            pred.extend(res)  # 把res中的元素添加到pred中
        pred = np.array(pred)  # 创建一个数组

        '''m = mean_absolute_error(test_y.flatten(), pred)
        r = np.sqrt(mean_squared_error(test_y.flatten(), pred))
        time = "%s" % datetime.now()
        mae = "%f" % m
        rmse = "%f" % r
        List = [time, mae, rmse]
        data = pd.DataFrame([List])
        data.to_csv('T:/assignment/GraduationProject/coding/FL/data/Health/mr.csv'
                    , mode='a', header=False, index=False)  # mae rmse'''

        print('mae:', mean_absolute_error(test_y.flatten(), pred), 'rmse:',  # flatten()返回一个一维数组
              np.sqrt(mean_squared_error(test_y.flatten(), pred)))  # mae：平均绝对误差 rmse：均方根误差