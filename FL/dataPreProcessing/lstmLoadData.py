# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/22 16:34
# @AUTHOR：WUWENBIN
# @FILENAME：lstmLoadData.py
# @SOFTNAME：PyCharm

import pandas as pd
import numpy as np

class lstmLoadData:
    def load_data():
        """
        :return: normalized dataframe
        """
        path = 'T:/assignment/GraduationProject/coding/FL/data/Health/LSTM/data/data.csv'
        df = pd.read_csv(path, encoding='gbk')
        columns = df.columns
        df.fillna(df.mean(), inplace=True)
        MAX = np.max(df[columns[1]])
        MIN = np.min(df[columns[1]])
        df[columns[1]] = (df[columns[1]] - MIN) / (MAX - MIN)

        return df, MAX, MIN