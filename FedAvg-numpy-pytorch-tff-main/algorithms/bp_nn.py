# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/27 15:36
@Author ：WUWENBIN
@File ：bp_nn.py

C：每次参与联邦聚合的clients数量占client总数的比例。C=1 代表所有成员参与聚合
E：两次联邦训练之间的本地训练的次数
B：client的本地的训练的batchsize
"""
import sys

import numpy as np
import pandas as pd

sys.path.append('../')
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import chain
from models import BP
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

clients = ['Client' + str(i) for i in range(1, 11)]
from args import args_parser


def load_data(file_name): # 读取csv格式文件
    df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/Health/Task 1/Client1_10/' + file_name + '.csv', encoding='gbk')
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


def nn_sequence(file_name, B): #全局模型
    print('data processing...')
    data = load_data(file_name) #加载经过load_data方法（上一方法 ）更新过的csv数据
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


def train(args, nn): #模型训练
    print('training...')
    train_x, train_y, test_x, test_y = nn_sequence(nn.file_name, args.B) #test_y = 全局模型
    nn.len = len(train_x)
    batch_size = args.B #batch_size每批数据量大小；args动态参数，任意数量个参数
    epochs = args.E #epochs 训练过程中数据将被轮多少次
    batch = int(len(train_x) / batch_size) #深度学习的优化算法
    for epoch in range(epochs):
        for i in range(batch):
            start = i * batch_size
            end = start + batch_size
            nn.forward_prop(train_x[start:end], train_y[start:end]) #正向传播计算
            nn.backward_prop(train_y[start:end]) #反向传播计算
        print('epoch:', epoch, ' error:', np.mean(nn.loss)) #第epoch轮的错误损失平均值
    return nn


def get_mape(x, y): #mape平均绝对百分比误差即误差占真实值的比，回归问题常用的评价标准
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))


def test(args, nn): #模型测试
    train_x, train_y, test_x, test_y = nn_sequence(nn.file_name, args.B) #test_y = 全局模型
    pred = []
    batch = int(len(test_y) / args.B) #深度学习优化算法
    for i in range(batch):
        start = i * args.B
        end = start + args.B
        res = nn.forward_prop(test_x[start:end], test_y[start:end]) #正向传播计算
        res = res.tolist()
        res = list(chain.from_iterable(res)) #将多个迭代器进行高效链接，将不同res拼起来并转换成列表
        #print('res=', res)
        pred.extend(res) #把res中的元素添加到pred中

    pred = np.array(pred) #创建一个数组
    print('mae:', mean_absolute_error(test_y.flatten(), pred), 'rmse:', #flatten()返回一个一维数组
          np.sqrt(mean_squared_error(test_y.flatten(), pred))) #mae：平均绝对误差 rmse：均方根误差
    print('mape', get_mape(pred,test_y.flatten()))


def main():
    args = args_parser()
    for client in clients:
        nn = BP(args, client)
        train(args, nn)
        test(args, nn)


if __name__ == '__main__':
    main()
