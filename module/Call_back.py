import os
import copy
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning.callbacks as pck

class MySaveCallback(Callback):
    TORCH_INF = torch_inf = torch.tensor(np.inf)
    MODE_DICT = {
        "min": (torch_inf, "min"),
        "max": (-torch_inf, "max"),
    }
    MONITOR_OP_DICT = {"min": torch.lt, "max": torch.gt}

    def __init__(self, args, monitor="mape", mode="min"):
        super(MySaveCallback, self).__init__()
        self.args = args
        self.monitor = monitor
        self.__init_monitor_mode(monitor, mode)
        self.best_epoch = 0

    def __init_monitor_mode(self, monitor, mode):
        self.best_value, self.mode = self.MODE_DICT[mode]

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx=0,
    ) -> None:
        monitor_op = self.MONITOR_OP_DICT[self.mode]
        metrics_dict = copy.copy(trainer.callback_metrics)
        monitor_value = metrics_dict.get(self.monitor, self.best_value)
        if monitor_op(monitor_value, self.best_value):
            self.best_value = monitor_value
            self.best_epoch = trainer.current_epoch


def load_callbacks(args):
    callbacks = []
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        strict=False,
        verbose=False,
        mode='max',
        min_delta=0.001
    ))
    callbacks.append(pck.ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        save_top_k=1,
        mode='max',
        save_last=True
    ))
    return callbacks
