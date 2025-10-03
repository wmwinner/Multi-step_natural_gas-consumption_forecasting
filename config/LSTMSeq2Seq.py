#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      LSTMSeq2SeqParams
@Author:    WuMian
@Date:      2024/8/7
@Email:     wumianwork@gmail.com 
"""


from config.BaseParams import BaseParams
from dataclasses import dataclass, field, asdict
from typing import *

@dataclass
class LSTMSeq2SeqParams(BaseParams):
    bidirectional: bool = field(
        default=False, metadata={"help": "确定LSTM模型是都是双向"}
    )
