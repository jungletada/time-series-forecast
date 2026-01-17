import os
import math

import yaml
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


plt.switch_backend('agg')


def seed_everything(seed=2026):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.backends.mps.manual_seed(seed)


def load_data_config(args):  
    config_path = args.data_config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.data_name not in config:
        raise ValueError(f'Dataset {args.data_name} not found in config file')
    config = config[args.data_name]
    args.data_type = config['data_type']
    args.root_path = config['root_path']
    args.data_path = config.get('data_path', None)
    args.selected_k = config.get('selected_k', 2)
    if config.get('target', None) is not None:
        args.target = config['target']
        
    if args.features == 'MS' or args.features == 'M':
        args.enc_in = config['channels']
        args.dec_in = config['channels']
        args.c_out = config['channels']
        
    elif args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
    
    return args


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

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
