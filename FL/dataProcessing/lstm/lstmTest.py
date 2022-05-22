# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/22 15:15
# @AUTHOR：WUWENBIN
# @FILENAME：lstmTest.py
# @SOFTNAME：PyCharm

from deepLearningCorrelation.buildLSTM import LSTM
from dataPreProcessing.lstmSequence import lstmSequence
from itertools import chain
import numpy as np
from universal.getMape import getMape
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from dataPreProcessing.lstmSequence import lstmSequence
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class lstmTest:
    def test(args, path):
        Dtr, Dte, m, n = lstmSequence.nn_seq(B=args.batch_size)
        pred = []
        y = []
        print('loading model...')
        input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
        output_size = args.output_size
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        # model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        model.load_state_dict(torch.load(path)['model'])
        model.eval()
        print('predicting...')
        for (seq, target) in tqdm(Dte):
            target = list(chain.from_iterable(target.data.tolist()))
            y.extend(target)
            seq = seq.to(device)
            with torch.no_grad():
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                pred.extend(y_pred)

        y, pred = np.array(y), np.array(pred)
        y = (m - n) * y + n
        pred = (m - n) * pred + n
        print('mape:', getMape.get_mape(y, pred))
        # plot
        x = [i for i in range(1, 151)]
        x_smooth = np.linspace(np.min(x), np.max(x), 900)
        y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
        plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

        y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
        plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
        plt.grid(axis='y')
        plt.legend()
        plt.show()