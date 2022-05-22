# !/user/bin/env python
# -*- coding:utf -8 -*-
# @TIME： 2022/5/22 16:14
# @AUTHOR：WUWENBIN
# @FILENAME：set_upSpeed.py
# @SOFTNAME：PyCharm

import os
import torch
import random
import numpy as np

class set_upSpeed:
    def setup_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True