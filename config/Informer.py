# !/usr/bin/env python
# -*- coding:utf-8 -*-
# %FileName :Informer.py
# %Time     :2025/6/7 17:58
# %Author   :wumian
# %Email    :wumianwork@gmail.com
from dataclasses import dataclass, field, asdict
from config.BaseParams import BaseParams
from typing import *

@dataclass
class InformerParams(BaseParams):
    # =============================================
    # 🤖 模型结构参数
    # =============================================
    model_name: str = field(default="Informer", metadata={"help": "模型名称"})

    lr: float = field(default=0.0001, metadata={"help": "学习率"})

    e_layers: int = field(default=2, metadata={"help": "Encoder编码层层数"})
    d_layers: int = field(default=1, metadata={"help": "Decoder解码层层数"})

    d_model: int = field(default=64, metadata={"help": "编码后的长度"})
    factor: int = field(default=3, metadata={"help": "attn factor"})
    d_ff: int = field(default=64, metadata={"help": "dimension of fcn"})

    top_k: int = field(default=5, metadata={"help": "for TimesBlock"})
    n_heads: int = field(default=8, metadata={"help": "num of heads"})

    dropout: float = field(default=0.1, metadata={"help": "Dropout 率"})

    # Informer模型的特有参数
    distil: bool = field(default=True, metadata={
        "help": "whether to use distilling in encoder, using this argument means not using distilling"})
