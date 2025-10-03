#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      __init__
@Author:    WuMian
@Date:      2024/9/19
@Email:     wumianwork@gmail.com 
"""

MODEL_DICT = {
    'MLP': [
        'BP',
        'DLinear',
        'FreTS',
        'Koopa',
        'TiDE',
        'TSMixer',
        'FiLM',
        'TimeMixer'
    ],
    'RNN': [
        'LSTM',
        "GRU",
        "RNN",
        "SegRNN"
    ],
    'Transformer': [
        'Autoformer',
        "Crossformer",
        "ETSformer",
        'FEDformer',
        "Informer",
        "iTransformer",
        'PatchTST',
        'Pyraformer',
        "Reformer",
        "Transformer",
        "HATL_Seq2Seq",
    ],
}


def get_model_type(model_name):
    for k, v in MODEL_DICT.items():
        if model_name in v:
            return k
    return model_name.keys()[0]
