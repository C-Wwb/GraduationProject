# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/12 13:26
@Author ：KI 
@File ：fedavg-numpy.py
@Motto：Hungry And Humble

"""
import numpy as np
import random
import copy
import sys

sys.path.append('../')
from algorithms.bp_nn import train, test #链接
from models import BP
from args import args_parser

clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]
#Task1_W_Zong1-10

class FedAvg: #平均联邦算法
    def __init__(self, args): #初始化
        self.args = args
        self.clients = args.clients
        self.nn = BP(args=args, file_name='server')
        self.nns = []
        # distribution，分配
        for i in range(self.args.K):
            s = copy.deepcopy(self.nn) #深拷贝（完全独立的两个模型）训练模型
            s.file_name = self.clients[i] #将训练模型对应不同客户
            self.nns.append(s)

    def server(self): #服务器端
        for t in range(self.args.r):
            print('round', t + 1, ':')
            m = np.max([int(self.args.C * self.args.K), 1])
            # sampling取样
            index = random.sample(range(0, self.args.K), m)
            # dispatch发送
            self.dispatch(index)
            # local updating本地更新
            self.client_update(index)
            # aggregation聚合
            self.aggregation(index)

        # return global model
        return self.nn

    def aggregation(self, index): #聚合函数
        # update w，更新w
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len
        w1 = np.zeros_like(self.nn.w1)
        w2 = np.zeros_like(self.nn.w2)
        w3 = np.zeros_like(self.nn.w3)
        w4 = np.zeros_like(self.nn.w4)
        for j in index:
            # normal
            w1 += self.nns[j].w1 * (self.nns[j].len / s)
            w2 += self.nns[j].w2 * (self.nns[j].len / s)
            w3 += self.nns[j].w3 * (self.nns[j].len / s)
            w4 += self.nns[j].w4 * (self.nns[j].len / s)
        # update server，更新服务端
        self.nn.w1, self.nn.w2, self.nn.w3, self.nn.w4 = w1, w2, w3, w4

    def dispatch(self, index):
        # distribute 分发
        for i in index:
            self.nns[i].w1, self.nns[i].w2, self.nns[i].w3, self.nns[
                i].w4 = self.nn.w1, self.nn.w2, self.nn.w3, self.nn.w4

    def client_update(self, index):  # update nn，更新全局模型
        for k in index:
            self.nns[k] = train(self.args, self.nns[k])

    def global_test(self): #全局测试
        model = self.nn
        c = clients_wind
        for client in c:
            model.file_name = client
            test(self.args, model)


def main():
    args = args_parser()
    fed = FedAvg(args)
    fed.server()
    fed.global_test()


if __name__ == '__main__':
    main()
