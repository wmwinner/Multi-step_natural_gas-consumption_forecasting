#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      timeUtils
@Author:    WuMian
@Date:      2024/8/10
@Email:     wumianwork@gmail.com 
"""
import pandas as pd
from dateutil import parser
from chinese_calendar import is_holiday, get_holiday_detail, is_in_lieu, is_workday
from datetime import datetime, timedelta, date
from typing import *


def date_to_str(data_date: Union[str, date, datetime, pd.Timestamp]) -> str:
    if type(data_date) == datetime or type(data_date) == pd.Timestamp or type(data_date) == date:
        return data_date.strftime('%Y-%m-%d')
    return data_date


def date_add(data_date: Union[str, date, datetime, pd.Timestamp], days: int) -> List[str]:
    if type(data_date) == datetime or type(data_date) == pd.Timestamp or type(data_date) == date:
        data_date = data_date.strftime('%Y-%m-%d')
    if days == 1:
        return [data_date]
    date_obj = datetime.strptime(data_date, '%Y-%m-%d')
    days_list = [timedelta(days=i) for i in range(1, days)]
    next_days_list = [date_obj] + [date_obj + i for i in days_list]
    predict_date_list = [i.strftime('%Y-%m-%d') for i in next_days_list]
    # 将结果转换回字符串格式并返回
    return predict_date_list

def date_sub(data_date: Union[str, datetime, pd.Timestamp, date], days: int) -> List[str]:
    if type(data_date) == datetime or type(data_date) == pd.Timestamp or type(data_date) == date:
        data_date = data_date.strftime('%Y-%m-%d')

    date_obj = datetime.strptime(data_date, '%Y-%m-%d')
    days_list = [timedelta(days=i) for i in range(1, days)]
    next_days_list = [date_obj] + [date_obj - i for i in days_list]
    predict_date_list = [i.strftime('%Y-%m-%d') for i in next_days_list]
    # 将结果转换回字符串格式并返回
    return predict_date_list


def disCal(data1: Union[str, datetime, pd.Timestamp, date], data2: Union[str, datetime, pd.Timestamp, date]) -> int:
    if type(data1) == datetime or type(data1) == pd.Timestamp or type(data1) == date:
        data1 = data1.strftime('%Y-%m-%d')
    if type(data2) == datetime or type(data2) == pd.Timestamp or type(data2) == date:
        data2 = data2.strftime('%Y-%m-%d')
    # 将字符串转换为datetime对象
    date1 = datetime.strptime(data1, "%Y-%m-%d")
    date2 = datetime.strptime(data2, "%Y-%m-%d")
    # 计算两个日期之间的天数差距
    days_difference = (date2 - date1).days
    return days_difference

def strToDate(data: Union[str, datetime, pd.Timestamp, date]) -> date:
    if type(data) == datetime or type(data) == pd.Timestamp:
        data = data.strftime('%Y-%m-%d')
    # 将字符串转换为datetime对象
    elif type(data) == date:
        return data
    data = datetime.strptime(data, "%Y-%m-%d").date()
    return data


def is_weekend(date_str) -> bool:
    date = pd.to_datetime(date_str)
    return date.dayofweek >= 5

def weekdays(date_str: Union[str, datetime, pd.Timestamp]) -> int:
    # 解析日期字符串为 datetime.date 对象
    if type(date_str) is str:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
    elif type(date_str) == datetime or type(date_str) == pd.Timestamp:
        date_obj = date_str.strftime('%Y-%m-%d')
        date_obj = datetime.strptime(date_obj, "%Y-%m-%d").date()
    elif type(date_str) == date:
        date_obj = date_str
    day_of_week = date_obj.weekday()+1
    return day_of_week
def isHoliday(date_str: Union[str, datetime, pd.Timestamp]) -> bool:
    if type(date_str) is str:
        dt = parser.parse(date_str)
    else:
        dt = date_str
    is_holi = is_holiday(dt) # 判断当天是否不上班[含调休、放假、周末]
    if is_holi:
        return 1
    else:
        return 0

def get_holiday_name(date_str: Union[str, datetime, pd.Timestamp]):
    is_holiday = isHoliday(date_str)
    if is_holiday:
        holiday_info = get_holiday_detail(date_str)
        if holiday_info[0]:  # 如果是法定假日
            holiday_name = holiday_info[1]
        else:  # 如果是调休工作日
            holiday_name = "调休工作日"
        if not holiday_name:
            return '普通周末'
        if holiday_name == 'Tomb-sweeping Day':
            holiday_name = "清明节"
        elif holiday_name == 'Labor Day':
            holiday_name = "劳动节"
        elif holiday_name == 'National Day':
            holiday_name = "国庆节"
        elif holiday_name == 'Chinese New Year':
            holiday_name = "春节"
        elif holiday_name == 'Dragon Boat Festival':
            holiday_name = "端午节"
        elif holiday_name == 'Mid-Autumn Festival':
            holiday_name = "中秋节"
        elif holiday_name == 'Lantern Festival':
            holiday_name = "元宵节"
        elif holiday_name == 'Spring Festival':
            holiday_name = "春节"
        return holiday_name
    else:
        return "非节假日"

def isRest(date_str: Union[str, datetime, pd.Timestamp]) -> bool:
    """
    判断是否是休息
    :param date_str:
    :return: 是返回1，否返回0
    """
    if type(date_str) is str:
        dt = parser.parse(date_str)
    else:
        dt = date_str

    is_work = is_workday(dt) # 判断当天是否不上班[含调休、放假、周末]
    if is_work:
        return 0
    else:
        return 1


def isSupplyGuarantee(date_str: Union[str, datetime, pd.Timestamp]) -> bool:
    """
    是否处于保供期,11.1-3.30
    :param date_str:
    :return:
    """
    if type(date_str) is str:
        dt = parser.parse(date_str)
        date_str = str(dt)
    else:
        dt = date_str
        date_str = str(date_str)
    year = dt.year
    before_year = year - 1
    next_year = year + 1

    start_time_1 = str(before_year) + '-11-01'
    end_time_1 = str(year) + '-03-30'
    start_time_2 = str(year) + '-11-01'
    end_time_2 = str(next_year) + '-03-30'
    if start_time_1 <= date_str and date_str <= end_time_1:
        return True
    else:
        if start_time_2 <= date_str and date_str <= end_time_2:
            return True
        return False