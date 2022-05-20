# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/27 19:51
# @AUTHOR：WUWENBIN
# @FILENAME：loadData.py
# @SOFTNAME：PyCharm

import sys

import numpy as np
import pandas as pd

sys.path.append('../')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class loadData:
    def load_data(file_name): # 读取csv格式文件
        df = pd.read_csv(os.path.dirname(os.getcwd()) + '/FL/data/Health/Task 1/Client1_10/' + file_name + '.csv', encoding='gbk')
        #os.path.dirname去掉文件名，返回目录
        #os.getcwd()返回当前工作目录
        #df是一个矩阵
        columns = df.columns #获取列名
        df.fillna(df.mean(), inplace=True)
        #fillna对缺失值进行填充，df.mean()按轴方向取平均，得到每列的平均值
        # 要把数据集中的缺失值填充起来，避免出错；这段是用每列的平均值将缺失值进行填充

        for i in range(3, 7): #3-7列是有意义值范围
            MAX = np.max(df[columns[i]]) #返回矩阵最大元素
            MIN = np.min(df[columns[i]]) #返回矩阵最小元素
            df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN) #更新df矩阵列值
        return df #返回该矩阵