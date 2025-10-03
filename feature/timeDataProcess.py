#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      DataProcess
@Author:    WuMian
@Date:      2023/9/26
@Email:     wumianwork@gmail.com 
"""
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import *
import copy



def data_feature_move(args, df: pd.DataFrame, feature_move=1) -> Optional[pd.DataFrame]:
    # 判断colum是否包括date等日期类型数据
    label_name = args.label
    # 将数据向前移动一行或者几行
    if feature_move:
        for column in args.data_columns:
            if column not in label_name:
                df[column] = df[column].shift(feature_move * -1)
    return df

def scaler(df: Optional[pd.DataFrame], is_s=None):
    if is_s:
        df = is_s.transform(df)
    else:
        # is_s = MinMaxScaler(feature_range=(-1, 1))
        is_s = StandardScaler()
        # 归一化数据集
        df = is_s.fit_transform(df)
    return df, is_s





