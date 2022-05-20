# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/27 20:54
# @AUTHOR：WUWENBIN
# @FILENAME：constructDataSet.py
# @SOFTNAME：PyCharm

from torch.utils.data import Dataset, DataLoader                  #数据集，加载数据

class MyDataset(Dataset):                                   #数据集
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

