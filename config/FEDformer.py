# !/usr/bin/env python
# -*- coding:utf-8 -*-
# %FileName :FEDformer.py
# %Time     :2025/6/11 03:18
# %Author   :wumian
# %Email    :wumianwork@gmail.com
from dataclasses import dataclass, field, asdict
from config.BaseParams import BaseParams
from typing import *


@dataclass
class FEDformerParams(BaseParams):
    # =============================================
    # 🤖 模型结构参数
    # =============================================
    model_name: str = field(default="FEDformer", metadata={"help": "模型名称"})
    lr: float = field(default=1e-3, metadata={"help": "学习率"})

    e_layers: int = field(default=2, metadata={"help": "Encoder编码层层数"})
    d_layers: int = field(default=2, metadata={"help": "Decoder解码层层数"})
    d_model: int = field(default=512, metadata={"help": "编码后的长度"})
    factor: int = field(default=3, metadata={"help": "attn factor"})
    d_ff: int = field(default=2048, metadata={"help": "output size"})
    top_k: int = field(default=5, metadata={"help": "for TimesBlock"})
    n_heads: int = field(default=8, metadata={"help": "num of heads"})

