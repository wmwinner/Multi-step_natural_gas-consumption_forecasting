#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      __init__
@Author:    WuMian
@Date:      2024/8/8
@Email:     wumianwork@gmail.com 
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "ml/config"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import dataclasses
from config.BaseParams import BaseParams
from config.Autoformer import AutoformerParams
from config.Crossformer import CrossformerParams
from config.FEDformer import FEDformerParams
from config.Informer import InformerParams
from config.TSMixer import TSMixerParams
from config.HATL_Seq2SeqParams import HATL_Seq2SeqParams
from config.LSTMSeq2Seq import LSTMSeq2SeqParams

from typing import List


def LoadConfig(model_type: str):
    model_type = model_type.lower()
    if model_type == "autoformer":
        return AutoformerParams
    elif model_type == "crossformer":
        return CrossformerParams
    elif model_type == "fedformer":
        return FEDformerParams
    elif model_type == "informer":
        return InformerParams
    elif model_type == "tsmixer":
        return TSMixerParams
    elif model_type == "hatl_seq2seq":
        return HATL_Seq2SeqParams
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def getModelParamsByName(model_name: str) -> dict:
    model_name = model_name.lower()

    return_params_name = ['task_type', 'features', 'seed', 'lr', 'train_epochs',  'batch_size', 'loss',
                          'train_size', 'val_size', 'test_size', 'output_test',
                          'seq_len', 'pred_len', 'label_len', 'dropout', 'activation', 'freq', 'embed'
                           ]
    if model_name in ["lstm", "gru"]:
        return_params_name.extend([
            'hidden_size', 'num_layers', 'bidirectional'
        ])
    elif model_name in ["autoformer", "crossformer", "fedformer", "informer"]:
        return_params_name.extend([
            'e_layers', 'd_layers', 'd_model', 'd_ff', 'top_k', 'factor'
        ])
    elif model_name == "lstmseq2seq":
        return_params_name.extend([
            'hidden_size', 'num_layers'])
    else:
        raise ValueError(f"未知的模型: {model_name}")

    param_class = LoadConfig(model_name)
    fields_dict = {f.name: f for f in dataclasses.fields(param_class)}

    params_fields_dict = {}

    for name in return_params_name:
        if name not in fields_dict:
            raise ValueError(f"Field '{name}' not found in dataclass")

        field_info = fields_dict[name]
        field_type = field_info.type
        default_value = field_info.default
        metadata = field_info.metadata or {}
        help_text = metadata.get("help", "")
        choices = metadata.get("choices", None)
        params_fields_dict[name] = {
                "type": field_type,
                "default": default_value,
                "help": help_text,
                "choices": choices
            }
    return params_fields_dict


def get_model_name_list() -> List[str]:
    model_name_list = ['Autoformer', 'Crossformer', 'DLinear', 'ETSformer', 'FiLM',
                       'FEDformer', 'LSTM', 'Transformer', 'Informer', 'PatchTST']
    return model_name_list


def getModelParams() -> dict:
    model_name_list = get_model_name_list()
    MODEL_PARAMS = {}
    for model_name in model_name_list:
        MODEL_PARAMS[model_name] = getModelParamsByName(model_name)
    return MODEL_PARAMS


def param_search(model_name: str):
    # 定义参数网格
    param_grid = {
        'LSTM':{
            "lr": [5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 1e-4],
            "d_model": [64, 128, 256, 512],
            "e_layers": [2, 3, 4],
            "dropout": [0.2, 0.3],
        },
        "Autoformer":{
            "lr": [5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 1e-4],
            "d_model": [256, 512, 1024, 2048],
            "e_layers": [2, 3, 4],
            "d_layers": [1, 2],
            "dropout": [0.2, 0.3],
        },
        "Crossformer": {
            "lr": [5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 1e-4],
            "d_model": [256, 512, 1024, 2048],
            "e_layers": [2, 3, 4],
            "d_layers": [1, 2],
            "dropout": [0.2, 0.3],
        },
        "FEDformer": {
            "lr": [5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 1e-4],
            "d_model": [256, 512, 1024, 2048],
            "e_layers": [2, 3, 4],
            "d_layers": [1, 2],
            "dropout": [0.2, 0.3],
        },
        "Informer": {
            "lr": [5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 1e-4],
            "d_model": [256, 512, 1024, 2048],
            "e_layers": [2, 3, 4],
            "d_layers": [1, 2],
            "dropout": [0.2, 0.3],
        },
        "TSMixer": {
            "lr": [5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 1e-4],
            "d_model": [256, 512, 1024],
            "e_layers": [2, 3, 4],
            "d_layers": [1, 2],
            "dropout": [0.2, 0.3],
        }
    }
    return param_grid[model_name]