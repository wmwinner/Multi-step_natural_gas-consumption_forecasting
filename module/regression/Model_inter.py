# -*- coding: utf-8 -*-
# @Author : wumian
# @Email : wumianwork@gmail.com
# @Time : 2023/7/15 10:28
import os
import sys
import pandas as pd
import torch
import pytorch_lightning as pl
from utils.metrics import metric
from utils.tools import inverse_transform, cal_accuracy
import numpy as np
import copy
from utils.losses import mape_loss, mase_loss, smape_loss, mse_loss, l1_loss, binary_cross_entropy, crossentropy_loss
from models.Transformer import *

now_path = os.path.dirname(__file__)



class Transformer_MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super(Transformer_MInterface, self).__init__()
        self.save_hyperparameters(kargs)
        self.model = self.load_model()
        print(self.model)
        self.configure_loss()
        self.min_mape = float('inf')

        self.test_predict_list = []
        self.test_label_list = []
        self.test_roll_predict_list = []
        self.test_roll_label_list = []
        self.predict_predict_list = []
        self.predict_result = []

    def load_model(self):
        # 绝对路径导入
        model_name = self.hparams.model_name.lower()
        print(model_name)
        if model_name == "autoformer":
            return Autoformer(self.hparams)
        elif model_name == "crossformer":
            return Crossformer(self.hparams)
        elif model_name == "fedformer":
            return FEDformer(self.hparams)
        elif model_name == "informer":
            return Informer(self.hparams)
        elif model_name == "tsmixer":
            return TSMixer(self.hparams)
        elif model_name == "hatl_seq2seq":
            return HATL_Seq2Seq(self.hparams)

    def configure_loss(self):
        loss = self.hparams.loss.lower()

        if loss == 'mse':
            self.loss_function = mse_loss()
        elif loss == 'l1':
            self.loss_function = l1_loss()
        elif loss == 'bce':
            self.loss_function = binary_cross_entropy()
        elif loss == 'mape':
            self.loss_function = mape_loss()
        elif loss == 'mase':
            self.loss_function = mase_loss()
        elif loss == 'smape':
            self.loss_function = smape_loss()
        elif loss == 'crossentropy':
            self.loss_function = crossentropy_loss()
        else:
            raise ValueError("Invalid Loss Type!")

    def configure_optimizers(self):
        # 检查对象self.hparams是否具有名为'weight_decay'的属性
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        elif self.hparams.optimizer == 'RAdam':
            optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        return optimizer


    def on_predict_start(self):
        self.predict_predict_list = []

    def on_test_epoch_start(self):
        self.test_predict_list = []
        self.test_label_list = []

        if self.hparams.test_rolling:
            # 获取测试数据加载器的第一个 batch
            test_loader = self.trainer.test_dataloaders
            steps = len(test_loader.dataset)
            first_batch = next(iter(test_loader))
            batch_x, batch_y, batch_x_mark, batch_y_mark = first_batch
            self.true_labels = batch_y  # 存储所有真实值
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x[-1:, :, :], batch_y[-1:, :, :], batch_x_mark[-1:, :, :], batch_y_mark[-1:, :, :]
            self.test_roll_predict_list = []
            self.test_roll_label_list = []
            # 初始化输入和标签
            self.current_input = batch_x.to(self.device)  # 初始输入
            # 开始递归预测
            for step in range(steps):
                B, _, C = self.current_input.shape
                dec_inp = torch.zeros((B, self.hparams.pred_len, C)).float().to(self.device)
                dec_inp = torch.cat([self.current_input[:, -self.hparams.label_len:, :], dec_inp], dim=1).float().to(self.device)
                with torch.no_grad():
                    outputs = self.model(self.current_input, None, dec_inp, None)
                # 保存预测和真实值
                self.test_roll_predict_list.append(copy.copy(outputs).cpu().numpy().reshape(-1, outputs.shape[-1]))

                true_value = self.true_labels[step, -self.hparams.pred_len:, :]
                self.test_roll_label_list.append(copy.copy(true_value).cpu().numpy().reshape(-1, true_value.shape[-1]))
                # 更新输入数据（滑动窗口）
                new_input = torch.cat([self.current_input[:, self.hparams.pred_len:, :], outputs], dim=1)
                self.current_input = new_input


    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        dec_inp = torch.zeros_like(batch_y[:, -self.hparams.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.hparams.label_len, :], dec_inp], dim=1).float()

        outputs = self.model(batch_x, None, dec_inp, None, mask=None)
        f_dim = -1 if self.hparams.features == 'MS' else 0
        outputs = outputs[:, -self.hparams.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.hparams.pred_len:, f_dim:]
        batch_y_mark = batch_y_mark[:, -self.hparams.pred_len:, f_dim:]
        loss = self.loss_function(batch_x, None, outputs, batch_y, batch_y_mark)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        B, _, C = batch_x.shape
        # decoder input
        dec_inp = torch.zeros((B, self.hparams.pred_len, C)).float().to(self.device)
        dec_inp = torch.cat([batch_x[:, -self.hparams.label_len:, :], dec_inp], dim=1).float()
        # encoder - decoder
        outputs = self.model(batch_x, None, dec_inp, None)

        outputs = outputs[:, -self.hparams.pred_len:, :]
        batch_y = batch_y[:, -self.hparams.pred_len:, :]
        outputs = outputs.cpu().numpy().reshape(-1, outputs.shape[-1])
        labels = batch_y.cpu().numpy().reshape(-1, batch_y.shape[-1])
        self.test_label_list.append(labels)
        self.test_predict_list.append(outputs)

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        B, _, C = batch_x.shape
        # decoder input
        dec_inp = torch.zeros_like(batch_x[:, -self.hparams.pred_len:, :]).float()
        dec_inp = torch.cat([batch_x[:, -self.hparams.label_len:, :], dec_inp], dim=1).float()

        # encoder - decoder
        outputs = self.model(batch_x, None, dec_inp, None)
        f_dim = -1 if self.hparams.features == 'MS' else 0
        outputs = outputs[:, -self.hparams.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.hparams.pred_len:, f_dim:]

        batch_y_mark = torch.ones(batch_y.shape).to(self.device)
        loss = self.loss_function(batch_x, None, outputs, batch_y, batch_y_mark)
        self.log("val_loss", loss)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        print(
            "batch_x.shape: ", batch_x.shape,
            "batch_y.shape: ", batch_y.shape,
            "batch_x_mark.shape: ", batch_x_mark.shape,
            "batch_y_mark.shape: ", batch_y_mark.shape
        )
        dec_inp = torch.zeros_like(batch_y[:, -self.hparams.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.hparams.label_len, :], dec_inp], dim=1).float()
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs = outputs.cpu().numpy().reshape(-1, outputs.shape[-1])
        print("predict_list: ", outputs)
        self.predict_predict_list.append(outputs)
        return outputs


    def on_test_epoch_end(self) -> None:
        f_dim = -1 if self.hparams.features == 'MS' else 0
        if self.hparams.test_rolling:
            test_predict = np.vstack(self.test_roll_predict_list)
            test_label = np.vstack(self.test_roll_label_list)
        else:
            test_predict = np.vstack(self.test_predict_list)
            test_label = np.vstack(self.test_label_list)

        if self.hparams.inverse:
            test_predict = inverse_transform(self.hparams, test_predict)
            test_label = inverse_transform(self.hparams, test_label)
        test_predict = test_predict[:, f_dim:]
        test_label = test_label[:, f_dim:]
        self.test_predict_list = test_predict
        self.test_label_list = test_label
        mae, mse, rmse, mape, mspe, r2, mape_list, mae_list = metric(pred=copy.copy(test_predict), true=copy.copy(test_label))

        if self.hparams.output_test:
            df_pred = pd.DataFrame(test_predict, columns=self.hparams.label)
            df_pred['数据类型'] = '预测值'
            df_pred[self.hparams.date_column] = self.hparams.date_list

            df_label = pd.DataFrame(test_label, columns=self.hparams.label)
            df_label['数据类型'] = '真实值'
            df_label[self.hparams.date_column] = self.hparams.date_list

            # 将预测值和真将预测值和真实值合并后保存。
            df = pd.concat([df_pred, df_label], axis=0)
            if not os.path.exists(self.hparams.output_path):
                os.mkdir(self.hparams.output_path)
            df.to_csv(f'./{self.hparams.output_path}/{self.hparams.user_name}.csv', index=False)
        metrics = {
            "MAE": mae,
            "MAPE": mape,
            "MSE": mse,
            "RMSE": rmse,
            "MSPE": mspe,
            "R2": r2,
        }
        # print('\n', metrics)
        self.log_dict(metrics)

    def on_predict_end(self):
        f_dim = -1 if self.hparams.features == 'MS' else 0
        test_predict = np.vstack(self.predict_predict_list)
        test_predict = inverse_transform(self.hparams, test_predict)
        self.predict_predict_list = test_predict[:, f_dim:]

