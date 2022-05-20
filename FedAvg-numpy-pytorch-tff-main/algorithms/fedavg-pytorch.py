# -*- coding:utf-8 -*-
"""
@Time: 2022/02/14 12:11
@Author: KI
@File: fedavg-pytorch.py
@Motto: Hungry And Humble

K，客户端数量，本文为10个，也就是10个地区。
C：选择率，每一轮通信时都只是选择C * K个客户端。
E：客户端更新本地模型的参数时，在本地数据集上训练E轮。
B：客户端更新本地模型的参数时，本地数据集batch大小为B
r：服务器端和客户端一共进行r轮通信。
clients：客户端集合。
type：指定数据类型，负荷预测or风功率预测。
lr：学习率。
input_dim：数据输入维度。
nn：全局模型。
nns： 客户端模型集合。
mape：预测评价指标，平均绝对百分比误差

在FedAvg的框架下：每一轮通信中，服务器分发全局参数到各个客户端，各个客户端利用本地数据训练相同的epoch，然后再将梯度上传到服务器进行聚合形成更新后的参数。
除了电力负荷数据以外，还有风功率数据，两个数据通过参数type指定：type == 'load’表示负荷数据，'wind’表示风功率数据。
"""
import copy
import random
import sys
from itertools import chain
from args import args_parser
import numpy as np
import torch

sys.path.append('../')
from sklearn.metrics import mean_absolute_error, mean_squared_error #mae：平均绝对误差 rmse：均方根误差
from models import ANN #神经网络 客户端采用pytorch搭建
from torch import nn #全局模型
from torch.utils.data import Dataset, DataLoader #数据集，加载数据
from algorithms.bp_nn import load_data

clients = ['Client' + str(i) for i in range(1, 11)]
#Client1-10


'''
PyTorch搭建LSTM实现时间序列预测（负荷预测）
'''
class MyDataset(Dataset): #数据集
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)



def nn_sequence(file_name, B): #B 每次本地更新的风功率数据量
    print('data processing...')
    data = load_data(file_name) #从路径加载待处理数据
    columns = data.columns #取出所有列的标头
    wind = data[columns[2]] #风功率是data中数据的第三列 TARGETVAR(0,1,2...)
    wind = wind.tolist() #将矩阵转换成列表
    data = data.values.tolist() #将数据全部转化为列表
    X, Y = [], [] #X=[当前时刻的year，month, hour, day, lowtemp, hightemp, 前一天当前时刻的负荷以及前23小时负荷]
    # Y=[当前时刻负荷]
    seq = []
    for i in range(len(data) - 30): #len（）返回对象中项目的数量，i < data中项目的数量-30
        train_seq = [] #训练数据集序列
        train_label = [] #训练数据集对应的标签
        for j in range(i, i + 24):
            train_seq.append(wind[j]) #append()方法将wind数据列表中的元素追加至训练数据集的尾部

        # 添加温度，湿度，气压等信息
        for c in range(3, 7):
            train_seq.append(data[i + 24][c])
        train_label.append(wind[i + 24])

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

class FedAvg:
    def __init__(self, args):
        self.args = args
        self.clients = args.clients
        self.nn = ANN(args, name='server').to(args.device) #gpu
        #self.nn，服务器端初始化的全局参数，由于服务器端不需要进行反向传播更新参数，因此不需要定义各个隐层以及输出。
        self.nns = []
        for i in range(args.K):
            temp = copy.deepcopy(self.nn) #深度拷贝
            temp.name = self.clients[i]
            self.nns.append(temp)

    def server(self): #服务器端
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling 样本
            m = np.max([int(self.args.C * self.args.K), 1]) #每一轮通信只选择c*k个客户端，并从中取出最大的一列
            index = random.sample(range(0, self.args.K), m)
            # index中存储m个0~10间的整数，表示被选中客户端的序号。
            # st random.sample（）截取列表中指定长度的随机字符。0-k个客户端里，随机取m个
            # 选取m个客户端后，并行操作更新本地wkt得到wkt+1，所有客户端更新结束后，将wkt+1传到服务器，服务器整合所有wkt+1，得到最新的全局参数wt+1
            # dispatch 分发
            self.dispatch(index)

            # local updating 本地更新
            self.client_update(index, t)

            # aggregation 聚合
            self.aggregation(index)

        return self.nn #返回全局模型

    '''
    把自己的数据集按照参数B分成若干个块，每一块大小都为B。
    对每一块数据，需要进行E轮更新：算出该块数据损失的梯度，然后进行梯度下降更新，得到新的本地w。
    更新完后w将被传送到中央服务器，服务器整合所有客户端计算出的w，得到最新的全局模型参数wt+1
    客户端收到服务器发送的最新全局参数模型参数，进行下一次更新。
    '''
    def aggregation(self, index): #参数聚合
        s = 0
        for j in index:
            # normal 原始论文中的方式，即根据样本数量来决定客户端参数在最终组合时所占比例。
            s += self.nns[j].len #客户端模型长度累加

        params = {} #将模型的weight和bias值清零
        for k, v in self.nns[0].named_parameters(): #通过named_parameters()获取所有的参数
            params[k] = torch.zeros_like(v.data) #torch.zeros_like():生成和括号内变量维度维度一致的全是零的内容。

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()
        # print(v.data)

    def dispatch(self, index): #分发，将更新后的参数分发给被选中的客户端
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()): #同时遍历客户端模型参数和全局模型参数
                old_params.data = new_params.data.clone() #使得旧参数更新为新参数

    def client_update(self, index, global_round):  # update nn 更新神经网络，客户端只需要利用本地数据来进行更新就行了
        for k in index:
            self.nns[k] = train(self.args, self.nns[k], global_round) #客户端更新

    def global_test(self): #全局测试
        model = self.nn
        model.eval()
        c = clients
        for client in c:
            model.name = client
            test(self.args, model)


def train(args, model, global_round): #训练
    model.train()
    Dtr, Dte = nn_sequence(model.name, args.B)
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

        print('epoch', epoch, ':', loss.item())

    return model


def test(args, model): #测试
    model.eval()
    Dtr, Dte = nn_sequence(model.name, args.B)
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
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred))) #mae：平均绝对误差 rmse：均方根误差


def main():
    args = args_parser()
    fed = FedAvg(args)
    fed.server()
    fed.global_test()


if __name__ == '__main__':
    main()

