# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/22 15:20
# @AUTHOR：WUWENBIN
# @FILENAME：getMape.py
# @SOFTNAME：PyCharm
import numpy as np
class getMape:
    def get_mape(x, y):  # mape平均绝对百分比误差即误差占真实值的比，回归问题常用的评价标准
        """
        :param x:true
        :param y:pred
        :return:MAPE
        """
        return np.mean(np.abs((x - y) / x))