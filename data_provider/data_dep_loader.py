import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
warnings.filterwarnings('ignore')

def merge_components(data_npy, k):
    """
    Merge IMFs into 3 components: High, Mid, Low frequencies.
    Input: [T, C, N_IMFS]
    Output: [T, C, 3]
    """
    T, C, N_IMFS = data_npy.shape
    
    # 边界保护：确保 k 在有效范围内
    if k < 0: k = 0
    if k >= N_IMFS: k = N_IMFS - 1
        
    print(f"   > Merging components with Pivot k={k} (Total IMFs={N_IMFS})")

    # 逻辑说明：
    # 0 ~ k-1 : High Frequency (Sum)
    # k       : Mid Frequency (Raw)
    # k+1 ~ end : Low Frequency (Sum)
    
    if k == 0: 
        # 特殊情况：第0个分量作为 Mid (通常不建议，除非IMFs非常少)
        # 此时 High 为空（或者把第0个既当High又当Mid），这里逻辑调整为：
        # comp1 (High) = 0 (或者全0矩阵，视具体需求，这里暂取第0个)
        # comp2 (Mid)  = 1
        # comp3 (Low)  = 2...end
        comp1 = data_npy[:, :, 0]
        comp2 = data_npy[:, :, 1]
        comp3 = np.sum(data_npy[:, :, 2:], axis=-1)
    elif k == N_IMFS - 1: 
        # 特殊情况：最后一个分量作为 Mid
        comp1 = np.sum(data_npy[:, :, :N_IMFS-2], axis=-1)
        comp2 = data_npy[:, :, N_IMFS-2]
        comp3 = data_npy[:, :, N_IMFS-1]
    else:
        # 标准情况
        # 注意 axis=-1 求和后维度会降低，需要 stack 恢复
        comp1 = np.sum(data_npy[:, :, :k], axis=-1)      # High
        comp2 = data_npy[:, :, k]                        # Mid
        comp3 = np.sum(data_npy[:, :, k+1:], axis=-1)    # Low

    decomp_data = np.stack([comp1, comp2, comp3], axis=-1) # [T, C, 3]
    return decomp_data

class Dataset_ETT_Decomposed(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, time_enc=0, freq='h', seasonal_patterns=None):
        self.args = args
        # size [seq_len, label_len, pred_len]
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
        self.scale = scale
        self.time_enc = time_enc
        self.root_path = root_path
        self.data_path = data_path
        
        # k parameter for component merging
        self.k = getattr(args, 'selected_k', 1) # 增加默认值防止报错
        
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
        
        if 'ETTm' in self.data_path:
            self.borders = self.border_map['ETTm']
        elif 'ETTh' in self.data_path:
            self.borders = self.border_map['ETTh']
        else:
            # 增加鲁棒性：如果是其他自定义数据集，可能需要手动指定 borders
            print(f"Warning: {self.data_path} not in [ETTh, ETTm], using default ETTh borders.")
            self.borders = self.border_map['ETTh']
            
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_name = os.path.splitext(self.data_path)[0]
        
        # 1. 构造文件名
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        # 你的 decomposition.py 生成的文件名应该包含 _scaled_cd
        npy_path = os.path.join(self.root_path, f"{base_name}_{current_flag_name}_sl{self.seq_len}_scaled_cd.npy")
        
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path) # [Split_Len, Total_Channels, N_IMFS]
        else:
            raise FileNotFoundError(f"Decomposed data not found: {npy_path}")
        
        # 2. 读取 CSV 获取原始数据和时间戳
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)
        
        # --- Feature Selection ---
        # df_raw columns: [date, col1, col2, ..., colN]
        # data_npy channels corresponds to [col1, col2, ..., colN] (index 0 to N-1)
        
        if self.features == 'M' or self.features == 'MS':
            # 取所有数据列
            cols_data = df_raw.columns[1:] 
            df_data = df_raw[cols_data]
            # NPY 不需要筛选，保持所有通道
            pass 
        elif self.features == 'S':
            # 只取 target 列
            if self.target not in df_raw.columns:
                 raise ValueError(f"Target {self.target} not found.")
            
            # 找到 target 在 "数据列" (去除date后) 中的索引
            data_cols = list(df_raw.columns[1:])
            target_idx = data_cols.index(self.target)
            
            df_data = df_raw[[self.target]]
            
            # 关键修正：NPY 数据也必须只保留 target 对应的通道
            # data_npy: [T, Total_C, K] -> [T, 1, K]
            data_npy = data_npy[:, target_idx:target_idx+1, :]

        # 3. 处理时间戳 (使用 .dt 加速)
        start_idx = self.borders['start'][self.set_type]
        end_idx = self.borders['end'][self.set_type]
        
        # 验证长度一致性 (非常重要)
        if len(data_npy) != (end_idx - start_idx):
            print(f"Warning: NPY len ({len(data_npy)}) != CSV split len ({end_idx - start_idx}).")
            # 通常以 NPY 长度为准，截断或调整 CSV 读取
            # 假设 NPY 是对的（因为是生成好的），我们调整 CSV 切片长度
            real_len = len(data_npy)
            end_idx = start_idx + real_len

        df_stamp = df_raw[['date']][start_idx:end_idx].copy() # Copy avoid warning
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.time_enc == 0:
            # 使用 .dt 访问器优化性能 (比 .apply 快很多)
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            
            if 'ETTm' in self.data_path:
                df_stamp['minute'] = df_stamp.date.dt.minute
                df_stamp['minute'] = df_stamp['minute'] // 15
                
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.time_enc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.args.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 4. 合并分量
        # Input: [T, C, N_IMFS] -> Output: [T, C, 3]
        data_processed = merge_components(data_npy, self.k)
        print(f"   > Loaded {current_flag_name}: X shape {data_processed.shape}")
            
        # 5. 标准化 (Scaling) - 仅针对 Target (Raw Data)
        # 注意：data_processed (X) 已经在 decomposition 阶段归一化过了，不需要再动
        if self.scale:
            # 使用 Train 集的数据 Fit Scaler
            train_s, train_e = self.borders['start'][0], self.borders['end'][0]
            train_raw_data = df_data[train_s:train_e]
            
            self.scaler.fit(train_raw_data.values)
            # 对当前需要的数据段进行 Transform
            # 注意：这里我们 transform 整个 CSV 段，然后切片
            raw_scaled = self.scaler.transform(df_data.values)
        else:
            raw_scaled = df_data.values

        # 修改后:
        self.data_decomp = data_processed            # 存储分解信号 [T, C, K]
        self.data_original = raw_scaled[start_idx:end_idx] # 存储原始信号 [T, C]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # ====================================================
        # 1. 构建输入 seq_x: [Seq_Len, C, K+1]
        # ====================================================
        # A. 获取分解部分 [Seq_Len, C, K]
        # 修改点：self.data_x -> self.data_decomp
        seq_x_decomp = self.data_decomp[s_begin:s_end, :, :]
        
        # B. 获取原始部分 [Seq_Len, C] -> 扩展为 [Seq_Len, C, 1]
        # 修改点：self.data_y -> self.data_original
        seq_x_original = self.data_original[s_begin:s_end, :]
        seq_x_original = seq_x_original[:, :, np.newaxis] 

        # C. 拼接: 原始信号在第0位 [Seq_Len, C, 1+K]
        seq_x = np.concatenate([seq_x_original, seq_x_decomp], axis=-1)
        
        # ====================================================
        # 2. 构建标签 seq_y: [Label_Len + Pred_Len, C, K+1]
        # ====================================================
        # A. 获取分解标签
        # 修改点：self.data_x -> self.data_decomp
        seq_y_decomp = self.data_decomp[r_begin:r_end, :, :]
        
        # B. 获取原始标签
        # 修改点：self.data_y -> self.data_original
        seq_y_original = self.data_original[r_begin:r_end, :]
        seq_y_original = seq_y_original[:, :, np.newaxis]
        
        # C. 拼接
        seq_y = np.concatenate([seq_y_original, seq_y_decomp], axis=-1)
        
        # ====================================================
        # 3. 时间戳 (不变)
        # ====================================================
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # 原 self.data_x 已更名为 self.data_decomp
        return len(self.data_decomp) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)