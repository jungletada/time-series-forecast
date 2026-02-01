import os
import warnings
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')

def merge_components(data_npy, k):
    print(f">>>>>>>>>>>> Merging components with k: {k}")
    if k is None:
        return data_npy
    # 处理分解数据的通道合并
    T, C, N_IMFS = data_npy.shape
    k = min(k, N_IMFS - 1)
    if k > 0 and k < N_IMFS - 1: 
        comp1 = np.sum(data_npy[:, :, :k], axis=-1)
        comp2 = data_npy[:, :, k]
        comp3 = np.sum(data_npy[:, :, k+1:], axis=-1)
    elif k == 0: # The first component
        comp1 = data_npy[:, :, 0]
        comp2 = data_npy[:, :, 1]
        comp3 = np.sum(data_npy[:, :, 2:], axis=-1)
    elif k == N_IMFS - 1: # The last component
        comp1 = np.sum(data_npy[:, :, :N_IMFS - 2], axis=-1)
        comp2 = data_npy[:, :, N_IMFS - 2]
        comp3 = data_npy[:, :, N_IMFS - 1]

    decomp_data = np.stack([comp1, comp2, comp3], axis=-1)
    return decomp_data

class Dataset_ETT_Decomposed(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, time_enc=0, freq='h', seasonal_patterns=None,):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.mnn = args.mnn
        self.use_residual = True
        use_mnn = getattr(args, 'use_mnn', 0)
        # mnn: 1 for True, 0 for False
        if use_mnn == 1:
            self.use_mnn = True
        else:
            self.use_mnn = False

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.target_idx = None
        self.scale = scale
        self.time_enc = time_enc
        self.root_path = root_path
        self.data_path = data_path
        
        # k parameter: IMFs 0~k-1 (High), k (Mid), k+1~end (Low)
        # 默认 k=3，意味着 IMF0, IMF1, IMF2 是高频，IMF3 是中频，剩下的低频
        # 根据实际 decomposition.py 的 max_imfs (例如15) 来调整这个 k
        self.k = self.args.selected_k
        
        # 这里的 borders 仅用于 CSV 时间戳的切分
        # 注意：必须与 decomposition.py 中的逻辑严格一致
        self.border_map = {
            'ETTh': {
                'start': [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len],
                'end':   [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            },
            'ETTm': {
                'start': [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len],
                'end':   [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            }
        }
        
        # 自动识别数据类型
        if 'ETTm' in self.data_path:
            self.borders = self.border_map['ETTm']
        elif 'ETTh' in self.data_path:
            self.borders = self.border_map['ETTh']
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
            
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_name = os.path.splitext(self.data_path)[0]
        
        # 1. 构造文件名 (确保 decomposition.py 生成的文件名包含 _scaled_cd)
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        npy_path = os.path.join(self.root_path, f"{base_name}_{current_flag_name}_sl{self.seq_len}_scaled_cd.npy")
        
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path)  # 注意：本身data_npy就已经是归一化后的数据
        else:
            raise FileNotFoundError(f"Decomposed data not found. Looked for {npy_path}")
        
        # Use for MNN ablation study
        pred_train_npy_path = os.path.join(self.root_path, f"pred_{base_name}_train_sl{self.seq_len}_scaled_cd.npy")
        if os.path.exists(pred_train_npy_path) and self.set_type == 0:
            print(f">>>>>>>>>>>>>>>>Loaded predicted data from {pred_train_npy_path}")
            data_pred_npy = np.load(pred_train_npy_path)
        else:
            data_pred_npy = None
        # data_npy Shape: [T, C, K_IMFS]
        # 2. 读取 CSV 以获取特征索引和时间戳
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)
        
        # --- Feature Selection (S or M) ---
        if self.features == 'M' or self.features == 'MS': # 排除第一列 date，取剩余所有
            cols_data = df_raw.columns[1:] # 对应的 Numpy 索引就是 0 到 C-1
            df_data = df_raw[cols_data]
            target_indices = list(range(len(cols_data))) 
        elif self.features == 'S': # 只取 target 列
            if self.target not in df_raw.columns:
                 raise ValueError(f"Target {self.target} not found in CSV columns.")
            # 找到 target 在 "数据列" (去除date后) 中的索引
            # df_raw.columns[1:] 对应 npy 的 Channel 维度
            data_cols = list(df_raw.columns[1:])
            df_data = df_raw[[self.target]]
            target_idx = data_cols.index(self.target)
            target_indices = [target_idx]
            self.target_idx = target_idx

        # 从 NPY 中筛选特征
        # data_npy: [T, Total_Channels, K] -> [T, Selected_Channels, K]
        data_npy = data_npy[:, target_indices, :]
        if data_pred_npy is not None and self.set_type == 0:
            data_pred_npy = data_pred_npy[:, target_indices, :]

        # 3. 处理时间戳 (Time Stamp), CSV 是全量的，需要切片
        start_idx = self.borders['start'][self.set_type]
        end_idx = self.borders['end'][self.set_type]
        s0, e0 = self.borders['start'][0], self.borders['end'][0]
        # s1, e1 = self.borders['start'][1], self.borders['end'][1]
        # s2, e2 = self.borders['start'][2], self.borders['end'][2]
        # 严格对齐检查
        if len(data_npy) != (end_idx - start_idx):
            # 这一步非常关键，如果 decomposition.py 的切分逻辑和这里的切分逻辑不一致，这里会报错
            print(f"Error: NPY len ({len(data_npy)}) != CSV split len ({end_idx - start_idx}). Truncating to shorter one.")
            exit(0)

        df_stamp = df_raw[['date']][start_idx:end_idx]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if 'ETTh' in self.data_path:
            if self.time_enc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.time_enc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
                data_stamp = data_stamp.transpose(1, 0)

        elif 'ETTm' in self.data_path:
            if self.time_enc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.time_enc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
                data_stamp = data_stamp.transpose(1, 0)
                
        # 4. 读取分量并且合并为3个分量 (Merge Components)
        data_processed = merge_components(data_npy, self.k)
        
        print(f">>>>>>>>>>>>> {current_flag_name} {self.use_mnn}: data_processed.shape: {data_processed.shape}")
            
        # 5. 标准化 (Scaling)
        if self.scale:
            # 加载 Train 数据计算 Mean/Std
            train_raw_data = df_data[s0:e0]
            self.scaler.fit(train_raw_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data_processed          # 输入是分解信号
        self.data_y = data[start_idx:end_idx] # 预测的是原始信号
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :, :]  # [T, C, K]
        seq_y = self.data_y[r_begin:r_end, :, :]  # [T, C]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 时间维度在索引 0
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        data: [Batch, T, C] 
        只对测试数据进行反归一化
        """
        return self.scaler.inverse_transform(data)
    