# !/usr/bin/env python
# -*- coding:utf-8 -*-
# %FileName :TSMixer.py
# %Time     :2025/6/10 12:50
# %Author   :wumian
# %Email    :wumianwork@gmail.com
from dataclasses import dataclass, field, asdict
from config.BaseParams import BaseParams
from typing import *

@dataclass
class TSMixerParams(BaseParams):
    # =============================================
    # 🤖 模型结构参数
    # =============================================
    model_name: str = field(default="TSMixer", metadata={"help": "模型名称"})
    lr: float = field(default=0.0001, metadata={"help": "学习率"})

    d_model: int = field(default=512, metadata={"help": "编码后的长度"})
    factor: int = field(default=3, metadata={"help": "attn factor"})
    d_ff: int = field(default=16, metadata={"help": "dimension of fcn"})
    top_k: int = field(default=5, metadata={"help": "for TimesBlock"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout 率"})