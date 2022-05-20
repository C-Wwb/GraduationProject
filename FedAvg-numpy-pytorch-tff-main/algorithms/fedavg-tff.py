# -*- coding:utf-8 -*-
"""
@Time: 2022/02/11 12:12
@Author: KI
@File: fedavg-tff.py
@Motto: Hungry And Humble
"""
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']='0' #cpu版本
import sys
sys.path.append('../')
from algorithms.bp_nn import load_data
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
from args import args_parser
args = args_parser()

nest_asyncio.apply()
tf.compat.v1.enable_v2_behavior()

clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]


# Data processing 客户数据处理
# 对于函数client_data(n, B, train_flag)，如果train_flag=True，返回客户端n的batch_size=B的训练集，否则返回测试集。
def client_data_wind(n, B, train_flag):
    print('data processing...')
    c = clients_wind
    data = load_data(c[n])
    if train_flag:
        data = data[0:int(len(data) * 0.9)]
    else:
        data = data[int(len(data) * 0.9):len(data)]

    label = data[data.columns[2]].values.tolist()
    data = data.values.tolist()
    X, Y = [], []
    for i in range(len(data) - 30):
        train_seq = []
        # train_label = []
        for j in range(i, i + 24):
            train_seq.append(label[j])
        for c in range(3, 7): #添加温度、湿度、气压等信息
            train_seq.append(data[i + 24][c])
        Y.append(label[i + 24])
        X.append(train_seq)

    X = tf.reshape(X, [len(X), -1])
    Y = tf.reshape(Y, [len(Y), -1])
    X = tf.data.Dataset.from_tensor_slices(X)
    Y = tf.data.Dataset.from_tensor_slices(Y)

    seq = tf.data.Dataset.zip((X, Y))
    seq = seq.batch(B, drop_remainder=True).shuffle(100).prefetch(B)
    # print(list(seq.as_numpy_iterator())[0])，batch_size=20

    return seq


# Wrap a Keras model for use with TFF.
# 构造ttf的keras模型
'''
keras_model：为联邦学习封装的Keras模型，该模型不能compile。
loss：损失函数。如果只提供一个损失函数，则所有模型都使用该损失函数；如果提供一个损失函数列表，则与各个客户端模型相互对应。这里选择MSE。
input_sec：指定模型的输入数据类型。input_spec必须是两个元素的复合结构，即x和y。如果作为列表提供，则必须按 [x, y]的顺序，如果作为字典提供，则键必须明确命名为“x”和“y”。本文是按照列表进行提供的。
loss_weights：可选项。如果loss为一个列表，那么就可以为每一个客户端的loss指定一个权重，最后求加权和。
metrics：可选项。这里选择了MAPE。
'''
def model_fn(): #最终返回的是一个tff.learning.Model，该模型将用于联邦学习
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    input_dim = args.input_dim

    model = tf.keras.models.Sequential([ #采用TensorFlow的keras模块来搭建了一个简单的神经网络
        tf.keras.layers.Dense(20, tf.nn.sigmoid, input_shape=(input_dim,),
                              kernel_initializer='zeros'),
        tf.keras.layers.Dense(20, tf.nn.sigmoid),
        tf.keras.layers.Dense(20, tf.nn.sigmoid),
        tf.keras.layers.Dense(1, tf.sigmoid)
    ])
    return tff.learning.from_keras_model(
        model,
        input_spec=train_data[0].element_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=metrics)


def FedAvg():
    # Simulate a few rounds of training with the selected client devices.
    trainer = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.08),
        # server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        # use_experimental_simulation_loop=True
    )
    state = trainer.initialize()
    for r in range(args.r):
        state, metrics = trainer.next(state, train_data)
        print('round', r + 1, 'loss:', metrics['train']['loss'])
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    for i in range(args.K):
        test_data = [client_data_wind(n, 20, train_flag=False) for n in range(i, i + 1)]
        # print('test:')
        test_metrics = evaluation(state.model, test_data)
        m = 'mean_absolute_error'
        print(str(test_metrics[m] / len(test_data[0])))


if __name__ == '__main__':
    train_data = [client_data_wind(n, 20, train_flag=True) for n in range(args.K)]
    FedAvg()
