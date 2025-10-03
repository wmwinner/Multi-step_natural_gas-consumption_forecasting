#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:
@File:      Model_train
@Author:    WuMian
@Date:      2024/8/9
@Email:     wumianwork@gmail.com
"""
import os
import numpy as np
import pandas as pd
import copy
import pytorch_lightning as pl
from module.regression.Data_inter import DInterface, Dataset_Inter
from pytorch_lightning.loggers import CSVLogger
from module.regression.Model_inter import Transformer_MInterface
from module.Call_back import load_callbacks
from typing import *
from feature.timeDataProcess import scaler as data_scaler
from config import LoadConfig, param_search
from sklearn.model_selection import ParameterGrid


def train_model(args, data_module) -> Union[Dict[str, Any], np.array, np.array, List, float]:
    model = Transformer_MInterface(**vars(args))
    checkpoint_callback = load_callbacks(args)

    logger = CSVLogger(args.lightning_log_path, name="train_log")
    trainer = pl.Trainer(
        devices=1,
        precision=args.precision,
        max_epochs=args.train_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=checkpoint_callback,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, data_module)
    val_loss = trainer.callback_metrics.get("val_loss")
    metrics_data = trainer.test(model=model, datamodule=data_module)
    return metrics_data[0], model.test_label_list, model.test_predict_list, args.date_list, val_loss

def train_factory(
        model_name,
        df: pd.DataFrame,
        update_params = None
):
    args = LoadConfig(model_name)()
    args.lightning_log_path = 'lightning_logs'

    if 'columns' in update_params:
        if type(update_params['columns']) == str:
            update_params['columns'] = eval(update_params['columns'])
    # 如果指定了标签列
    if 'label' in update_params:
        if type(update_params['label']) == str:
            update_params['label'] = eval(update_params['label'])
    for key, value in update_params.items():
        setattr(args, key, value)

    if not df.empty and args.freq == 'd':
        df[args.date_column] = pd.to_datetime(df[args.date_column])
    df.sort_values(by=args.date_column, inplace=True)
    args.date_list = df[args.date_column].tolist()

    data_columns = copy.copy(args.columns)
    if args.date_column in data_columns:
        data_columns.remove(args.date_column)
    # 如果是多序列预测单序列
    if len(args.label) == len(data_columns):
        args.features = 'M'
    else:
        if len(args.label) == 1:
            args.features = 'MS'
        else:
            raise ValueError(
                f"预测类型是{args.features}, 表示多序列预测单序列，因此label 维度必须为1， 请检查label参数或者")
    if args.features == 'MS':
        col_index = data_columns.index(args.label[0])
        del data_columns[col_index]
        data_columns.append(args.label[0])
    args.data_columns = data_columns

    args.dec_in = len(args.data_columns)
    args.c_out = len(args.data_columns)
    args.enc_in = len(args.data_columns)

    label_index = []
    for i in args.label:
        label_index.append(args.data_columns.index(i))
    args.label_index = label_index


    df_data = df[args.data_columns]
    df, scaler = data_scaler(copy.copy(df_data))
    args.scaler = scaler
    train_dataset = Dataset_Inter(args, df,  args.date_list, 'train')
    val_dataset = Dataset_Inter(args, df,  args.date_list, 'val')
    test_dataset = Dataset_Inter(args, df,  args.date_list, 'test')
    data_module = DInterface(trainset=train_dataset, valset=val_dataset, testset=test_dataset, args=args)

    if args.grid_search:
        # 定义参数网格
        param_grid = param_search(args.model_name)
        # 网格搜索
        best_score = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            for key, value in params.items():
                if hasattr(args, key):
                    setattr(args, key, value)

            metrics_data, test_label_list, test_predict_list, date_list, val_loss = train_model(args,
                                                                                                        data_module)
            if val_loss is not None and val_loss < best_score:
                best_score = val_loss
                best_params = params
                print(f"New best params: {params} with val_loss={val_loss:.4f}")
        if best_params:
            for key, value in best_params.items():
                if hasattr(args, key):  # 可选检查，防止设置不存在的属性
                    setattr(args, key, value)
            metrics_data, test_label_list, test_predict_list, date_list, val_loss = train_model(args, data_module)
    else:
        metrics_data, test_label_list, test_predict_list, date_list, val_loss = train_model(args, data_module)
    return metrics_data, test_label_list, test_predict_list, date_list