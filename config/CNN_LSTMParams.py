#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      CNN_LSTMParams
@Author:    WuMian
@Date:      2024/8/7
@Email:     wumianwork@gmail.com 
"""
import os
import sys

from config.BaseParams import BaseParams
from dataclasses import dataclass, field, asdict
from typing import *


@dataclass
class LSTMSeq2SeqParams(BaseParams):
    kernel_size: (int, int) = field(
        default=(2, 2) , metadata={"help": "隐藏层神经元数量"}
    )
