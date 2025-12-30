import os
import time
import warnings
import numpy as np
import pandas
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')

class Exp_Dep_Short_Term_Forecasting(Exp_Basic):
    def __init__(self, args, logger):
        # 定义分量名称，用于日志打印
        self.comp_names = ['High', 'Mid', 'Low']
        self.num_components = len(self.comp_names)

        super(Exp_Dep_Short_Term_Forecasting, self).__init__(args)
        self.logger = logger
        self.logger.info(f'Initializing Exp_Dep_Short_Term_Forecasting.')
        self.logger.info(f'Number of components: {self.num_components}')

    def _build_model(self):
        if self.args.data_type == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2 * pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]

        models = []
        for i in range(self.num_components):
            model = self.model_dict[self.args.model].Model(self.args).float()
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            models.append(model)
        return nn.ModuleList(models)
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
    
    
