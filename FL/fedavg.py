# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/27 20:05
# @AUTHOR：WUWENBIN
# @FILENAME：fedavg.py
# @SOFTNAME：PyCharm


import copy
import random
import sys
import numpy as np
import torch
import torch.utils.data.distributed #android

sys.path.append('../')
from deepLearningCorrelation.buildNeuralNetwork import ANN #神经网络 客户端采用pytorch搭建
from universal.args import args
from dataProcessing.fedavg.fedTest import fedTest
from dataProcessing.fedavg.fedTrain import fedTrain

clients = ['Client' + str(i) for i in range(1, 11)]
#Client1-10

class FedAvg:
    def __init__(self, args):
        self.args = args
        self.clients = args.clients
        self.nn = ANN(args, name='server').to(args.device) #搭建客户端模型
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
            print(index)

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

    def client_update(self, index, global_round):
        for k in index:
            self.nns[k] = fedTrain.train(self.args, self.nns[k], global_round)

    def global_test(self): #全局测试
        model = self.nn
        model.eval()
        c = clients
        for client in c:
            model.name = client
            fedTest.test(self.args, model)

def main():
    fed = FedAvg(args.args_parser())
    fed.server()
    fed.global_test()

if __name__ == '__main__':
    main()
