# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/27 19:46
# @AUTHOR：WUWENBIN
# @FILENAME：fedavgSequence.py
# @SOFTNAME：PyCharm

import sys
import torch

sys.path.append('../')
from dataPreProcessing.loadData import loadData
from torch.utils.data import Dataset, DataLoader #数据集，加载数据
from dataProcessing.constructDataSet import MyDataset

class structuralSequence:
    def nn_sequence(file_name, B): #B 每次本地更新的风功率数据量
        print('data processing...')
        data = loadData.load_data(file_name) #从路径加载待处理数据
        columns = data.columns #取出所有列的标头
        rmssd = data[columns[2]] #风功率是data中数据的第三列 TARGETVAR(0,1,2...)
        rmssd = rmssd.tolist() #将矩阵转换成列表
        data = data.values.tolist() #将数据全部转化为列表
        X, Y = [], [] #X=[当前时刻的year，month, hour, day, lowtemp, hightemp, 前一天当前时刻的负荷以及前23小时负荷]
        # Y=[当前时刻负荷]
        seq = []
        for i in range(len(data) - 30): #len（）返回对象中项目的数量，i < data中项目的数量-30
            train_seq = [] #训练数据集序列
            train_label = [] #训练数据集对应的标签
            for j in range(i, i + 24):
                train_seq.append(rmssd[j]) #append()方法将wind数据列表中的元素追加至训练数据集的尾部

            # 添加温度，湿度，气压等信息
            for c in range(3, 7):
                train_seq.append(data[i + 24][c])
            train_label.append(rmssd[i + 24])

            train_seq = torch.FloatTensor(train_seq).view(-1) #torch.FloatTensor（）默认生成32位浮点数，.view（-1）使之成为一行数据
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
            #print(seq[:5])

        Dtr = seq[0:int(len(seq) * 0.8)] #训练集
        Dte = seq[int(len(seq) * 0.8):len(seq)] #测试集

        train_len = int(len(Dtr) / B) * B
        test_len = int(len(Dte) / B) * B
        Dtr, Dte = Dtr[:train_len], Dte[:test_len]

        train = MyDataset(Dtr)
        test = MyDataset(Dte)

        Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0) #每次通过dataloader从训练集中取出一个数据
        Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0) #每次通过dataloader从测试集中取出一个数据

        return Dtr, Dte
'''
上面代码用了DataLoader来对原始数据进行处理，最终得到了batch_size=B的数据集Dtr和Dte，Dtr为训练集，Dte为测试集。
'''