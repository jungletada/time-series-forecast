import os
import math
import copy
import yaml
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from argparse import Namespace

plt.switch_backend('agg')


def seed_everything(seed=2026):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.backends.mps.manual_seed(seed)


def build_model_args(base_args: Namespace, model_cfg: dict) -> Namespace:
    """
    从全局 base_args 拷贝一份，然后用 model_cfg 覆盖/追加字段，
    返回一个可以像 args 一样用的 model_args。
    """
    model_args = copy.deepcopy(base_args)
    for k, v in model_cfg.items():
        setattr(model_args, k, v)
    return model_args

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def override_args_with_yaml(args, parser, config: dict):
    """
    用 YAML 配置覆盖 args 中仍处于“默认值”的字段。
    规则：命令行 > YAML > argparse 默认值
    """
    for key, value in config.items():
        if hasattr(args, key):
            # argparse 中定义过的参数
            default = parser.get_default(key)
            current = getattr(args, key)

            # 只有当这个参数仍然是默认值时，才使用 YAML 覆盖
            # ——如果用户在命令行里改过，就不会动它
            if current == default:
                setattr(args, key, value)
        else:
            # argparse 里没定义，但 YAML 想额外挂一些属性（如 data_type、root_path）
            # 也可以直接附加在 args 上
            setattr(args, key, value)

    return args


def apply_model_config(args, parser):
    if not getattr(args, 'model_config', None):
        return args

    model_cfg = load_yaml_config(args.model_config)
    args = override_args_with_yaml(args, parser, model_cfg)
    return args


def apply_data_config(args, parser):
    if not getattr(args, 'data_config', None):
        print('No data config found')
        return args

    with open(args.data_config, 'r') as f:
        all_cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.data_name not in all_cfg:
        raise ValueError(f'Dataset {args.data_name} not found in config file')

    data_cfg = all_cfg[args.data_name]

    # 1）先用通用规则：命令行 > YAML > 默认
    args = override_args_with_yaml(args, parser, data_cfg)

    # 2）再处理你原来根据 features 设置 enc_in/dec_in/c_out 的逻辑，
    #    但同样只在它们还等于默认值时才覆盖（命令行仍然优先）
    if 'selected_k' in data_cfg:
        args.selected_k = getattr(args, 'selected_k', data_cfg['selected_k'])
    
    if args.features in ['MS', 'M']:
        channels = data_cfg['channels']
        for name in ['enc_in', 'dec_in', 'c_out']:
            if hasattr(args, name):
                default = parser.get_default(name)
                current = getattr(args, name)
                if current == default:
                    setattr(args, name, channels)

    elif args.features == 'S':
        for name in ['enc_in', 'dec_in', 'c_out']:
            if hasattr(args, name):
                default = parser.get_default(name)
                current = getattr(args, name)
                if current == default:
                    setattr(args, name, 1)

    return args


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def adjust_learning_rate(optimizer, epoch, base_lr, args):
    # # 支持标量和列表形式的 learning_rate
    # if isinstance(args.learning_rate, (list, tuple)):
    #     if len(args.learning_rate) == 0:
    #         raise ValueError("args.learning_rate is an empty list.")
    #     base_lr = float(args.learning_rate[0])
    # else:
    #     base_lr = float(args.learning_rate)

    # 根据 lradj 选择不同的学习率策略
    if args.lradj == 'type1':
        lr_adjust = {epoch: base_lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {
            epoch: base_lr if epoch < 3
            else base_lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: base_lr / 2.0 * (1.0 + math.cos(epoch / args.train_epochs * math.pi))
        }
    else:
        # 未知的 lradj 策略，则不调整
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def visual(true, preds, horizon_len=None, name='./pic/test.pdf', title=None):
    """
    Results visualization
    """
    plt.figure(figsize=(7, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = plt.gca()
    ax.tick_params(labelsize=21)
    ax.set_facecolor('#F9F8F7')
    plt.gcf().patch.set_facecolor('#FFFFFF')
    plt.plot(
        true,
        label='GroundTruth',
        linewidth=2.4,
        linestyle='-',
        color='#1E90FF',
        alpha=1.0
    )
    if horizon_len is not None:
        total_len = len(preds)
        split = min(int(horizon_len), total_len)
        plt.plot(
            np.arange(0, split),
            preds[:split],
            label='_nolegend_',
            linewidth=2.4,
            linestyle='--',
            color='#1E90FF',
            alpha=1.0
        )

        plt.plot(
            np.arange(split, total_len),
            preds[split:],
            label='Prediction',
            linewidth=2.2,
            linestyle='--',
            color='#FF4500',
            alpha=0.9
        )
    else:
        plt.plot(
            preds,
            label='Prediction',
            linewidth=2.2,
            linestyle='--',
            color='#FF4500',
            alpha=1.0
        )
    
    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
    plt.legend(
        frameon=True,
        ncol=1,
        fontsize=15,
        loc='upper left',
        framealpha=0.6,
        facecolor='#ffffff',
        edgecolor='#cccccc'
    )
    ax.text(
        0.5,
        -0.08,
        title if title is not None else '',
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=22,
        fontweight='bold',
        color='#333333'
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(name, bbox_inches='tight', dpi=150)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
