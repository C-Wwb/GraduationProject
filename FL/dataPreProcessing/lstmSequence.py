# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/22 15:08
# @AUTHOR：WUWENBIN
# @FILENAME：lstmSequence.py
# @SOFTNAME：PyCharm
from dataPreProcessing.lstmLoadData import lstmLoadData
import torch
from dataProcessing.constructDataSet import MyDataset
from torch.utils.data import Dataset, DataLoader


class lstmSequence:
    def nn_seq(B):
        print('data processing...')
        data, m, n = lstmLoadData.load_data()
        load = data[data.columns[1]]
        load = load.tolist()
        load = torch.FloatTensor(load).view(-1)
        data = data.values.tolist()
        seq = []
        for i in range(len(data) - 24):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                train_seq.append(load[j])
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq).view(-1)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        # print(seq[:5])

        Dtr = seq[0:int(len(seq) * 0.7)]
        Dte = seq[int(len(seq) * 0.7):len(seq)]

        train_len = int(len(Dtr) / B) * B
        test_len = int(len(Dte) / B) * B
        Dtr, Dte = Dtr[:train_len], Dte[:test_len]

        train = MyDataset(Dtr)
        test = MyDataset(Dte)

        Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
        Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

        return Dtr, Dte, m, n