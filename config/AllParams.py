# !/usr/bin/env python
# -*- coding:utf-8 -*-
# %FileName :AllParams.py
# %Time     :2025/6/19 16:51
# %Author   :wumian
# %Email    :wumianwork@gmail.com

from dataclasses import dataclass, field, asdict
from config.LSTM import LSTMParams
from config.Transformer import TransformerParams
from typing import *

@dataclass
class AllParams(TransformerParams):
    # =============================================
    # 🤖 模型结构参数
    # =============================================

  # Informer模型的特有参数
    distil: bool = field(default=True, metadata={
        "help": "whether to use distilling in encoder, using this argument means not using distilling"})

    # TimeMixer模型特有的参数
    channel_independence: int = field(default=1, metadata={
        "help": "0: channel dependence 1: channel independence for FreTS model"})

    decomp_method: str = field(default='moving_avg', metadata={
        "help": "method of series decompsition, only support moving_avg or dft_decomp"})
    moving_avg: int = field(default=25, metadata={"help": "window size of moving average"})
    use_norm: int = field(default=1, metadata={"help": "whether to use normalize; True 1 False 0"})
    down_sampling_method: str = field(default='avg',
                                      metadata={"help": "down sampling method", "choices": ['avg', 'max', 'conv']})

    # LSTM 模型特有的参数
    bidirectional: bool = field(default=False, metadata={"help": "是否使用双向 LSTM/GRU"})

