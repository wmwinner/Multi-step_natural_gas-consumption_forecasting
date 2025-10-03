# !/usr/bin/env python
# -*- coding:utf-8 -*-
# %FileName :dataprocess.py
# %Time     :2025/6/13 13:35
# %Author   :wumian
# %Email    :wumianwork@gmail.com
import torch
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from typing import *
from sklearn.preprocessing import MinMaxScaler
from typing import *
import copy

def scaler(df: Optional[pd.DataFrame], is_s=None):
    if is_s:
        df = is_s.transform(df)
    else:
        is_s = MinMaxScaler(feature_range=(-1, 1))
        # 归一化数据集
        df = is_s.fit_transform(df)
    return df, is_s