#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Project:   
@File:      MondelParams
@Author:    WuMian
@Date:      2023/11/7
@Email:     wumianwork@gmail.com 
"""
from dataclasses import dataclass, field, asdict
from typing import *


@dataclass
class DataParams:
    """
    调控中心特有参数
    """
    id: int = field(default=0, metadata={"help": "用户的id"})
    user_name: str = field(default="user", metadata={"help": "预测对象名称"})
    user_type: int = field(default=0, metadata={"help": "预测对象类型：0-管道，1-区域，2-重点用户"})
    freq: str = field(default='d', metadata={
        "help": "freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
        "choices": ["s", "t", "h", "d", "b", "w", "m", "y", "other"]})
    time_step: int = field(default=1, metadata={"help": "时间间隔长度，默认为1"})



@dataclass
class BaseParams(DataParams):
    """
    所有训练、模型、数据、日志、任务等配置参数统一管理
    """
    # =============================================
    # 🧪 训练相关参数
    # =============================================
    seed: int = field(default=1234, metadata={"help": "随机种子"})
    lr: float = field(default=1e-4, metadata={"help": "学习率"})
    train_epochs: int = field(default=400, metadata={"help": "训练的总 epoch 数"})
    batch_size: int = field(default=256, metadata={"help": "训练时的 batch size"})
    check_val_every_n_epoch: int = field(default=20, metadata={"help": "每多少个 epoch 验证一次"})
    log_every_n_steps: int = field(default=20, metadata={"help": "每多少步记录一次日志"})
    precision: int = field(default=32, metadata={"help": "训练精度", "choices": [16, 32]})
    weight_decay: float = field(default=1e-5, metadata={"help": "L2 正则化系数"})
    patience: int = field(default=5, metadata={"help": "早停法的耐心值"})
    optimizer : str = field(default="AdamW", metadata={"help": "优化器名称", "choices": ["Adam", "RAdam", "SGD", "RMSprop", "AdamW"]})
    test_rolling: bool = field(default=False, metadata={"help": "是否使用滚动窗口进行预测"})
    grid_search: bool = field(default=False, metadata={"help": "是否启用网格搜索优化超参数"})

    # =============================================
    # 📊 数据集划分与预处理参数
    # =============================================patience
    train_size: float = field(default=0.8, metadata={"help": "训练集占比"})
    val_size: float = field(default=0.1, metadata={"help": "验证集占比"})
    test_size: float = field(default=0.1, metadata={"help": "测试集占比"})
    columns: Optional[List[str]] = field(default=None, metadata={"help": "要使用的列名列表"})
    date_column: Optional[str] = field(default='date', metadata={"help": "时间列名"})
    data_columns: Optional[List[str]] = field(default=None, metadata={"help": "输入特征列名列表"})
    label: Optional[List[str]] = field(default_factory=list, metadata={"help": "标签列名"})
    label_index: Optional[List[int]] = field(default_factory=list, metadata={"help": "标签在 columns 中的索引"})
    seq_len: int = field(default=64, metadata={"help": "输入序列长度（历史时间步数）"})
    pred_len: int = field(default=32, metadata={"help": "预测未来的时间步数"})
    label_len: int = field(default=6, metadata={"help": "用于解码器起始 token 的长度"})
    feature_move: int = field(default=1, metadata={"help": "移位步长（滑动窗口步长）"})
    time_feature: bool = field(default=False, metadata={"help": "是否添加日期特征（如星期）"})
    use_week: bool = field(default=True, metadata={"help": "是否使用星期作为特征"})
    interp: bool = field(default=True, metadata={"help": "是否对缺失值进行插值"})
    interval: int = field(default=0, metadata={"help": "标签之间的间隔"})

    # =============================================
    # 🧩 任务与用户定义参数
    # =============================================
    task_type: str = field(
        default="短期预测",
        metadata={"help": "任务类型", "choices": ["短期预测", "长期预测", "分类", "异常检测"]}
    )
    task_name: str = field(
        default="normal",
        metadata={"help": "任务类型", "choices": ["normal", "work_day", "holiday"]}
    )
    features: str = field(default='M',
                        metadata={"help": 'forecasting task; M:多变量 预测 多变量, S:单变量预测单变量, MS:多变量 预测 单变量',
                                  "choices": ['M', 'S', 'MS']})
    embed: str = field(default='timeF', metadata={"help": "时间特征编码方式", "choices": ['timeF', 'fixed', 'learned']})
    use_dtw: bool = field(default=False, metadata={"help": "the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)"})

    # =============================================
    # 💾 日志与输出路径
    # =============================================
    lightning_log_path: str = field(default="logdir/lightning_logs", metadata={"help": "Lightning 日志路径"})
    output_path: str = field(default="output", metadata={"help": "输出文件路径"})
    output_test: bool = field(default=True, metadata={"help": "是否导出测试集预测结果"})
    # label: str = field(default="", metadata={"help": "预测结果表头名称"})

    # =============================================
    # 🛠️ 其他高级参数
    # =============================================
    loss: str = field(default="MSE", metadata={"help": "损失函数", "choices": ["MSE", "BCELoss", "MAPE", "MASE", "SMAPE"]})
    devices: int = field(default=0, metadata={"help": "GPU 设备编号，0 表示自动选择"})
    num_workers: int = field(default=0, metadata={"help": "CPU 工作线程数"})
    seasonal_patterns: str = field(default='Daily', metadata={"help": "季节模式（如 Monthly）", "choices": ["Hourly", "Daily", "Weekly", "Monthly", "Yearly", "Quarterly"]})
    augmentation_ratio: int = field(default=0, metadata={"help": "数据增强倍数"})
    use_label: bool = field(default=True, metadata={"help": "是否将标签作为输入的一部分"})
    inverse: bool = field(default=False, metadata={"help": "是否转换输出为原始数据"})
    lradj: str = field(default='type1', metadata={"help": "adjust learning rate"})

    # ================================================
    # 模型参数
    # ================================================
    enc_in: int = field(default=3, metadata={"help": "encoder input size"})
    dec_in: int = field(default=2, metadata={"help": "decoder input size"})
    c_out: int = field(default=3, metadata={"help": "output size（输出维度）"})

    e_layers: int = field(default=2, metadata={"help": "Encoder编码层层数"})
    d_layers: int = field(default=1, metadata={"help": "Decoder解码层层数"})
    d_model: int = field(default=512, metadata={"help": "编码后的长度"})

    activation : str = field(default="gelu", metadata={"help": "激活函数", "choices": ["relu", "gelu","sigmoid"]})
    dropout: float = field(default=0.1, metadata={"help": "Dropout 率"})
    output_attention: bool = field(default=False, metadata={"help": "是否输出注意力权重（仅编码器）"})
