# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/11 9:05
# @AUTHOR：WUWENBIN
# @FILENAME：pyDraw.py
# @SOFTNAME：PyCharm

import csv
import pandas as pd
import numpy as np
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

with open('/data/Health/client1.csv'
        , 'r') as c:
    #print(c.read())
    r = csv.reader(c)
    step, loss, mae = [],[],[]
    index = 0
    for i in r:
        if(index != 0):
            step.append(i[1])
            loss.append(i[2])
            mae.append(i[3])
        index += 1
    list = ['step', 'loss', 'mae']
    lists = {};
    lists["step"], lists["loss"], lists["mae"] = step, loss, mae

x = range(0, 100, 1)
#plt.ylim(0, 0.8)
line2,=plt.plot(x, loss, 'r--', linewidth = 0.8)
line3,=plt.plot(x, mae, 'b-', linewidth = 0.8)
ll = plt.legend([line3, line2], ["mae", "loss"], loc = 'upper right')

plt.grid(axis="y", linestyle = '--')
plt.text(60000, 0.13, 'loss', fontdict={'size':9, 'color': 'red'})
plt.text(70000, -0.03, 'mae', fontdict={'size':9, 'color': 'blue'})
plt.ylabel("loss", fontsize=11)
plt.xlabel("step", fontsize=11)

plt.savefig('T:/assignment/GraduationProject/coding/FL/data/Health/client2.jpg'
            , dpi = 1200)
plt.rcParams['figure.dpi']=900
plt.show()

