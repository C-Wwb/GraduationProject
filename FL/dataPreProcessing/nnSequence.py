# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/28 21:36
# @AUTHOR：WUWENBIN
# @FILENAME：nnSequence.py
# @SOFTNAME：PyCharm

import sys
from dataPreProcessing.loadData import loadData
sys.path.append('../')
import numpy as np

class nnSequence:
    def nn_sequence(file_name, B): #全局模型
        print('data processing...')
        data = loadData.load_data(file_name) #加载经过load_data方法（上一方法 ）更新过的csv数据
        columns = data.columns #取出所有的列
        wind = data[columns[2]] #风功率是data中数据的第二列
        wind = wind.tolist() #将矩阵转换成列表
        data = data.values.tolist() #将数据全部转化为列表
        X, Y = [], []
        for i in range(len(data) - 30): #len（）返回对象中项目的数量，i < data中项目的数量-30
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                train_seq.append(wind[j]) #append()方法向列表末尾追加元素。

            for c in range(3, 7):
                train_seq.append(data[i + 24][c])
            train_label.append(wind[i + 24])
            X.append(train_seq)
            Y.append(train_label)

        X, Y = np.array(X), np.array(Y) #创建x，y数组
        train_x, train_y = X[0:int(len(X) * 0.8)], Y[0:int(len(Y) * 0.8)] #训练集
        test_x, test_y = X[int(len(X) * 0.8):len(X)], Y[int(len(Y) * 0.8):len(Y)] #测试集

        train_len = int(len(train_x) / B) * B #训练集长度
        test_len = int(len(test_x) / B) * B #测试集长度
        train_x, train_y, test_x, test_y = train_x[:train_len], train_y[:train_len], test_x[:test_len], test_y[:test_len]

        # print(len(train_x))
        return train_x, train_y, test_x, test_y #返回训练集和测试集

    def get_mape(x, y): #mape平均绝对百分比误差即误差占真实值的比，回归问题常用的评价标准
        """
        :param x:true
        :param y:pred
        :return:MAPE
        """
        return np.mean(np.abs((x - y) / x))