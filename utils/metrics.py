import numpy as np
from typing import *
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def R2(pred, true):
    print(pred.shape, true.shape)
    r2 = r2_score(true, pred)
    return r2

def MAE(pred: np.array, true: np.array):
    return np.mean(np.abs(pred - true))


def MSE(pred: np.array, true: np.array):
    return np.mean((pred - true) ** 2)


def RMSE(pred: np.array, true: np.array):
    return np.sqrt(MSE(pred, true))


def MAPE(pred: np.array, true: np.array):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred: np.array, true: np.array):
    return np.mean(np.square((pred - true) / true))


def MAPE_list(pred: np.array, true: np.array):
    return abs(true - pred) / abs(true)

def MAE_list(pred: np.array, true: np.array):
    return abs(true - pred)


def metric(pred: Union[list, np.array], true: Union[list, np.array]):
    if type(true) == list:
        true = np.array(true)
    if type(pred) == list:
        pred = np.array(pred)
    # mae = MAE(pred, true)
    # mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    mae = mean_absolute_error(true, pred)
    try:
        mape = mean_absolute_percentage_error(true, pred)
    except ValueError as e:
        mape = None
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = RMSE(pred, true)
    mspe = MSPE(pred, true)
    mape_list = MAPE_list(pred, true)
    mae_list = MAE_list(pred, true)
    # r2 = R2(pred, true)
    true = true.reshape(-1)
    pred = pred.reshape(-1)
    df = pd.DataFrame({'真实值': true.tolist(), '预测值': pred.tolist()})
    # df.to_csv('预测结果对比.csv', index=False)
    return mae, mse, rmse, mape, mspe, r2, mape_list, mae_list
