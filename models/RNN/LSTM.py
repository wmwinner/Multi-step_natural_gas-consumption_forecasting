#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      LSTM
@Author:    WuMian
@Date:      2023/9/26
@Email:     wumianwork@gmail.com 
"""
import torch.nn as nn
import torch
from typing import *
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.configs = configs
        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.output_size = configs.c_out
        self.pred_len = configs.pred_len
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=configs.bidirectional)
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.linear2 = nn.Linear(self.hidden_size * 4, self.output_size)
        # self.linear2 = nn.Linear(self.hidden_size * 4, configs.output_size*configs.pred_len)
        self.dropout = nn.Dropout(configs.dropout)

    def attention_net(self, x):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :return:
        """
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_size]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score  # [batch, seq_len, hidden_size]
        context = torch.sum(scored_x, dim=1)  # [batch, hidden_size]
        return context

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # out: [batch_size, seq_len, hidden_size*num_directions]
        # h_n: [num_directions*num_layers, batch_size, hidden_size]
        # out[:, -1, :]
        out, (h_n, c_n) = self.lstm(x_enc)
        if self.configs.output_attention:
            out = self.attention_net(out)
        # else:
        #     out = out[:, -1, :]
        out = self.linear1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out[:, -self.pred_len:, :]
        # return out  # torch.Size([128, 1])


