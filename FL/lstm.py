# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/22 15:42
# @AUTHOR：WUWENBIN
# @FILENAME：lstm.py
# @SOFTNAME：PyCharm

from universal.lstmArgs import lstmArgs
from universal.set_upSpeed import set_upSpeed
from dataProcessing.lstm.lstmTrain import lstmTrain
from dataProcessing.lstm.lstmTest import lstmTest


set_upSpeed.setup_seed(20)
LSTM_PATH = 'T:/assignment/GraduationProject/coding/FL/data/Health/LSTM/model//Univariate-SingleStep-LSTM.pkl'

if __name__ == '__main__':
    args = lstmArgs.us_args_parser()
    lstmTrain.train(args, LSTM_PATH)
    lstmTest.test(args, LSTM_PATH)