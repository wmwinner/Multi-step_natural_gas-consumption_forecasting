#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:
@File:      models
@Author:    WuMian
@Date:      2024/4/17
@Email:     wumianwork@gmail.com
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from feature.timefeatures import time_features
import torch
import warnings
from models import get_model_type
from utils.timeUtils import isHoliday
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')


class DInterface(pl.LightningDataModule):
    def __init__(self,
                 trainset=None,
                 valset=None,
                 testset=None,
                 predset=None,
                 args=None):
        super().__init__()
        self.num_workers = args.num_workers
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.predset = predset

        self.batch_size = args.batch_size

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)



class Dataset_Inter(Dataset):
    def __init__(self, args, df, date_list=None, flag='train'):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2,  'pred': 3}
        self.set_type = type_map[flag]
        self.freq = args.freq
        self.timeenc = 0 if args.embed != 'timeF' else 1
        self.df = df
        self.date_list = date_list
        if flag == 'pred':
            self.__read_pred_data__()
        else:
            self.__read_data__()
    def __get_test_date(self, begin_index, end_index):
        date_list = self.args.date_list[begin_index:end_index]
        new_date_list = []
        for i in range(len(date_list) - self.seq_len-self.pred_len + 1 - self.args.interval):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end + self.args.interval
            r_end = r_begin + self.pred_len
            date_mid = date_list[r_begin:r_end]
            new_date_list.append(date_mid)
        self.args.date_list = [item for sublist in new_date_list for item in sublist]

    def __read_data__(self):
        # 使用间隔标签的时候pred_len必须为1，不然无法进行数据处理
        data_len = len(self.df)
        # data = self.df
        border1s = [0, int(data_len * self.args.train_size) - self.seq_len, int(data_len * (self.args.train_size + self.args.val_size)) - self.seq_len]
        border2s = [int(data_len * self.args.train_size), int(data_len * (self.args.train_size + self.args.val_size)), data_len - 1]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = self.df[border1:border2, :]
        self.data_y = self.df[border1:border2, :]


        if get_model_type(self.args.model_name) in ['RNN']:
            self.label_len = 0

        data_stamp = time_features(pd.to_datetime(self.date_list), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

        if self.set_type == 2:
            self.__get_test_date(border1, border2)

    def __read_pred_data__(self):
        self.data_x = self.df[-1 * self.args.seq_len:, :]
        # 这里的data_y其实没有用，仅仅是为了保证模型可以接受
        self.data_y = np.zeros((self.seq_len + self.pred_len+self.args.interval, len(self.args.label)))
        if get_model_type(self.args.model_name) in ['RNN']:
            self.label_len = 0
        data_stamp = time_features(pd.to_datetime(self.date_list), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp[-1 * (self.seq_len + self.pred_len):]

    def __getitem__(self, index):
        # 为了让数据加载下标能够适应不同的模型需求，对LSTM这类模型设置 self.label_len=0
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len + self.args.interval
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return torch.tensor(seq_x).float(), torch.tensor(seq_y).float(), torch.tensor(seq_x_mark).float(), torch.tensor(seq_y_mark).float()

    def __len__(self):
        return len(self.data_y) - self.seq_len-self.pred_len + 1 - self.args.interval


class HDataset_Inter(Dataset):
    def __init__(self, args, df, date_list=None, flag='train'):
        self.args = args
        self.data_X, self.data_Y = self.data_split(df, date_list)
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2,  'pred': 3}
        self.set_type = type_map[flag]
        self.freq = args.freq
        self.df = df
        self.date_list = date_list

        if flag == 'pred':
            self.__read_pred_data__()
        else:
            self.__read_data__()

    # 数据拆分，如果要预测假期的用气量那么对数据进行拆分
    def data_split(self, df: np.array, date_list:list):
        data_X = []
        data_Y = []

        if get_model_type(self.args.model_name) in ['RNN']:
            label_len = 0
        else:
            label_len = self.args.label_len
        for i in range(self.args.seq_len, len(date_list)):
            s_begin = i
            s_end = s_begin + self.args.seq_len
            r_begin = s_end - label_len
            r_end = r_begin + label_len + self.args.pred_len
            if self.args.task_type == 'holiday':
                if isHoliday(date_list[i]):
                    data_X.append(df[s_begin:s_end])
                    if len(self.args.label) == 1 and get_model_type(self.args.model_name) in ['RNN']:
                        data_Y = df[r_begin:r_end, self.args.label_index[0]]
                    else:
                        data_Y = df[r_begin:r_end, self.args.label_index]
            else:
                if not isHoliday(date_list[i]):
                    data_X.append(df[s_begin:s_end])
                    if len(self.args.label) == 1 and get_model_type(self.args.model_name) in ['RNN']:
                        data_Y = df[r_begin:r_end, self.args.label_index[0]]
                    else:
                        data_Y = df[r_begin:r_end, self.args.label_index]
        return data_X, data_Y

    def __read_data__(self):
        data_len = len(self.data_X)
        border1s = [0, int(data_len * self.args.train_size) - self.seq_len, int(data_len * (self.args.train_size + self.args.val_size)) - self.seq_len]
        border2s = [int(data_len * self.args.train_size), int(data_len * (self.args.train_size + self.args.val_size)), data_len - 1]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = self.data_X[border1:border2]
        self.data_y = self.data_Y[border1:border2]
        if get_model_type(self.args.model_name) in ['RNN']:
            self.label_len = 0


        data_stamp = time_features(pd.to_datetime(self.date_list), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            print('='*100)
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __read_pred_data__(self):
        self.data_x = self.df[-1:]
        self.data_y = self.df[-1:]

        data_stamp = time_features(pd.to_datetime(self.date_list), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp[-1 * (self.seq_len + self.pred_len):]

    def __getitem__(self, index):
        # 为了让数据加载下标能够适应不同的模型需求，对LSTM这类模型设置 self.label_len=0
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark = self.data_stamp[index]
        seq_y_mark = self.data_stamp[index]
        return torch.tensor(seq_x).float(), torch.tensor(seq_y).float(), torch.tensor(seq_x_mark).float(), torch.tensor(seq_y_mark).float()

    def __len__(self):
        return len(self.data_y)