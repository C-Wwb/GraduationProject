# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/4/27 19:39
# @AUTHOR：WUWENBIN
# @FILENAME：buildNeuralNetwork.py
# @SOFTNAME：PyCharm

from torch import nn

class ANN(nn.Module): #客户端采用pytorch搭建本地模型
    def __init__(self, args, name):
        super(ANN, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.sigmoid = nn.Sigmoid()
        #神经网络激活函数，输出范围是（0,1）
        self.fc1 = nn.Linear(args.input_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 1)
    # input_dim 输入数据维度，这一行数据有多少个元素组成
    # nn.Linear（）用来设置网络的全连接层，全连接的输入输出一般都为二维张量——nn.Linear（in_features，out_features）
    # in_features 输入张量的形状决定
    # out_features 输出张量的形状决定，也代表了该全连接层的神经元个数
    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x
