import os
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')


class Dataset_ETT_Decomposed(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, time_enc=0, freq='h', 
                 seasonal_patterns=None):
        use_mnn = getattr(args, 'use_mnn', 0)
        if use_mnn == 1:
            self.test_mnn = True
        else:
            self.test_mnn = False
        # size [seq_len, label_len, pred_len]
        self.args = args
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
        
        # k parameter: IMFs 0~k-1 (High), k (Mid), k+1~end (Low)
        # 默认 k=3，意味着 IMF0, IMF1, IMF2 是高频，IMF3 是中频，剩下的低频
        # 根据实际 decomposition.py 的 max_imfs (例如10) 来调整这个 k
        self.k = getattr(self.args, 'selected_k', 2) 
        
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
            self.freq = 't'
        elif 'ETTh' in self.data_path:
            self.borders = self.border_map['ETTh']
            self.freq = 'h'
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
            
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_name = os.path.splitext(self.data_path)[0]
        
        # 1. 构造文件名 (确保 decomposition.py 生成的文件名包含 _cd)
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        # 尝试加载带 seq_len 的文件名 (更安全)，如果找不到则加载默认的
        npy_path = os.path.join(self.root_path, f"{base_name}_{current_flag_name}_sl{self.seq_len}_cd.npy")
        
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path)
        else:
            raise FileNotFoundError(f"Decomposed data not found. Looked for {npy_path}")

        # data_npy Shape: [T, C, K_IMFS]
        # 注意：这里的 data_npy 已经是切分好的片段，不要再做时间切片！

        # 2. 读取 CSV 以获取特征索引和时间戳
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)
        
        # --- Feature Selection (S or M) ---
        # 我们需要知道保留哪些 Column Index
        if self.features == 'M' or self.features == 'MS': # 排除第一列 date，取剩余所有
            cols_data = df_raw.columns[1:] # 对应的 Numpy 索引就是 0 到 C-1
            target_indices = list(range(len(cols_data))) 
        elif self.features == 'S': # 只取 target 列
            if self.target not in df_raw.columns:
                 raise ValueError(f"Target {self.target} not found in CSV columns.")
            # 找到 target 在 "数据列" (去除date后) 中的索引
            # df_raw.columns[1:] 对应 npy 的 Channel 维度
            data_cols = list(df_raw.columns[1:])
            target_idx = data_cols.index(self.target)
            target_indices = [target_idx]
        
        # 从 NPY 中筛选特征
        # data_npy: [T, Total_Channels, K] -> [T, Selected_Channels, K]
        data_npy = data_npy[:, target_indices, :]

        # 3. 处理时间戳 (Time Stamp), CSV 是全量的，需要切片
        start_idx = self.borders['start'][self.set_type]
        end_idx = self.borders['end'][self.set_type]
        
        # 严格对齐检查
        if len(data_npy) != (end_idx - start_idx):
            # 这一步非常关键，如果 decomposition.py 的切分逻辑和这里的切分逻辑不一致，这里会报错
            print(f"Error: NPY len ({len(data_npy)}) != CSV split len ({end_idx - start_idx}). Truncating to shorter one.")
            exit(0)
            # min_len = min(len(data_npy), end_idx - start_idx)
            # data_npy = data_npy[:min_len]
            # end_idx = start_idx + min_len

        df_stamp = df_raw[['date']][start_idx:end_idx]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.time_enc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.time_enc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 4. 合并分量 (Merge Components)
        # data_npy is [T, C, K_IMFS]
        k = self.k
        T, C, N_IMFS = data_npy.shape
        
        # 确保 k 不越界
        k = min(k, N_IMFS - 1)
        
        # Comp 1: High Freq (0 to k-1)
        comp1 = np.sum(data_npy[:, :, :k], axis=-1) if k > 0 else np.zeros((T, C))
        # Comp 2: Mid Freq (k)
        comp2 = data_npy[:, :, k]
        # Comp 3: Low Freq (k+1 to end)
        comp3 = np.sum(data_npy[:, :, k+1:], axis=-1) if k+1 < N_IMFS else np.zeros((T, C))
        
        # Stack -> [T, C, 3]
        data_processed = np.stack([comp1, comp2, comp3], axis=-1)

        if self.set_type == 2 and self.test_mnn:
            mnn_npy_path = os.path.join(self.root_path, f"pred_{base_name}_test_sl{self.seq_len}_cd.npy")
            data_mnn = np.load(mnn_npy_path)
            data_mnn = data_mnn.reshape(-1, 1, 3)
            assert data_mnn.shape[0] == data_processed.shape[0]
            length, num_channels, num_imfs = data_processed.shape
            data_pad = np.zeros((length, num_channels-1, num_imfs))
            data_processed = np.concatenate([data_pad, data_mnn], axis=1)

        # 5. 标准化 (Scaling)
        if self.scale:
            # 加载 Train 数据计算 Mean/Std
            train_npy_path = os.path.join(self.root_path, f"{base_name}_train_sl{self.seq_len}_cd.npy")
            
            # 读取 Train 并筛选特征
            train_npy = np.load(train_npy_path) # [T_train, Total_C, K]
            train_npy = train_npy[:, target_indices, :] # [T_train, Selected_C, K]
            
            # 还原为原始信号值来 Fit Scaler
            train_sum = np.sum(train_npy, axis=-1) # [T_train, C]
            self.scaler.fit(train_sum)
            
            # 获取 scaler 的参数，形状适配 [1, C, 1] 以便广播
            # mean: [C], scale: [C]
            mean = self.scaler.mean_.reshape(1, C, 1)
            scale = self.scaler.scale_.reshape(1, C, 1)
            
            # --- 关键 Scaling 逻辑 ---
            # data_processed: [T, C, 3] -> [96, C, 3] -> [336, C, 3]
            # 0: High, 1: Mid, 2: Low
            
            # 1. High Freq & Mid Freq: 仅除以 scale (假设它们是围绕0波动的)
            data_processed[:, :, 0:2] = data_processed[:, :, 0:2] / scale
            
            # 2. Low Freq (Trend): 减去 Mean 并除以 scale (它承担了基准偏移)
            data_processed[:, :, 2:3] = (data_processed[:, :, 2:3] - mean) / scale

        # 6. 转置为模型需要的格式
        # 通常 Time-Series-Library 是 [T, C]
        # 这里为了保持 __getitem__ 方便，我们先存为 [3, T, C]
        self.data_x = data_processed.transpose(2, 0, 1) # [3, T, C]
        self.data_y = data_processed.transpose(2, 0, 1) # [3, T, C]
        self.data_stamp = data_stamp

        # Augmentation (仅对 Train 有效)
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            # 注意：augmentation 库通常期望输入是 [T, C]，这里是 [3, T, C]
            # 直接调用可能会报错，取决于 utils.augmentation 的实现。
            # 这里建议先忽略，或者需要对每一层分别做 augmentation
            pass 

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # self.data_x: [3, T, C]
        # 保持第一维 (Component) 不变，切片第二维 (Time)
        seq_x = self.data_x[:, s_begin:s_end, :] 
        seq_y = self.data_y[:, r_begin:r_end, :]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 时间维度在索引 1
        return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        data: [Batch, 3, T, C] 或者 [Batch, T, C] (如果模型已经求和了)
        """
        # 如果输入是分开的 3 个分量，先求和
        if data.ndim == 4 and data.shape[1] == 3:
            data = data.sum(dim=1) # [Batch, T, C]
            
        # 调用 scaler 还原
        # 注意: scaler 期望输入是 [Batch * T, C] 或者 numpy
        # 这里简单封装，假设 data 是 Tensor 或 Numpy
        if hasattr(data, 'cpu'): data = data.cpu().numpy()
        
        shape = data.shape
        # Flatten time dims
        if data.ndim == 3:
            data = data.reshape(-1, shape[-1])
            
        inverse_data = self.scaler.inverse_transform(data)
        return inverse_data.reshape(shape)
    
class Dataset_Custom_Decomposed(Dataset):
    def __init__(
        self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, time_enc=0, freq='h', seasonal_patterns=None, split_ratio=(0.7, 0.2)):
        # [seq_len, label_len, pred_len]
        use_mnn = getattr(args, 'use_mnn', 0)
        if use_mnn == 1:
            self.test_mnn = True
        else:
            self.test_mnn = False
        self.args = args
        self.split_ratio = split_ratio
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # initialize
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.time_enc = time_enc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        base_name = os.path.splitext(self.data_path)[0]
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        # 1. 构造文件名 (确保 decomposition.py 生成的文件名包含 _cd)
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        # 加载带 seq_len 的文件名
        npy_path = os.path.join(self.root_path, f"{base_name}_{current_flag_name}_sl{self.seq_len}_cd.npy")
        
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path)
        else:
            raise FileNotFoundError(f"Decomposed data not found. Looked for {npy_path}")

        # data_npy Shape: [T, C, K_IMFS]
        # 注意：这里的 data_npy 已经是切分好的片段，不要再做时间切片！

        # 2. 读取 CSV 以获取特征索引和时间戳
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw)  * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        self.borders = {
            'start': [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            'end':   [num_train, num_train + num_vali, len(df_raw)]
        }
        # --- Feature Selection (S or M) ---
        # 我们需要知道保留哪些 Column Index
        if self.features == 'M' or self.features == 'MS': # 排除第一列 date，取剩余所有
            cols_data = df_raw.columns[1:] # 对应的 Numpy 索引就是 0 到 C-1
            target_indices = list(range(len(cols_data))) 
        elif self.features == 'S': # 只取 target 列
            if self.target not in df_raw.columns:
                 raise ValueError(f"Target {self.target} not found in CSV columns.")
            # 找到 target 在 "数据列" (去除date后) 中的索引
            # df_raw.columns[1:] 对应 npy 的 Channel 维度
            data_cols = list(df_raw.columns[1:])
            target_idx = data_cols.index(self.target)
            target_indices = [target_idx]
        
        # 从 NPY 中筛选特征
        # data_npy: [T, Total_Channels, K] -> [T, Selected_Channels, K]
        data_npy = data_npy[:, target_indices, :]

        # 3. 处理时间戳 (Time Stamp), CSV 是全量的，需要切片
        start_idx = self.borders['start'][self.set_type]
        end_idx = self.borders['end'][self.set_type]
        
        # 严格对齐检查
        if len(data_npy) != (end_idx - start_idx):
            # 这一步非常关键，如果 decomposition.py 的切分逻辑和这里的切分逻辑不一致，这里会报错
            print(f"Error: NPY len ({len(data_npy)}) != CSV split len ({end_idx - start_idx}). Truncating to shorter one.")
            exit(0)

        df_stamp = df_raw[['date']][start_idx:end_idx]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.time_enc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.time_enc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 4. 合并分量 (Merge Components)
        # data_npy is [T, C, K_IMFS]
        k = self.k
        T, C, N_IMFS = data_npy.shape
        
        # 确保 k 不越界
        k = min(k, N_IMFS - 1)
        
        # Comp 1: High Freq (0 to k-1)
        comp1 = np.sum(data_npy[:, :, :k], axis=-1) if k > 0 else np.zeros((T, C))
        # Comp 2: Mid Freq (k)
        comp2 = data_npy[:, :, k]
        # Comp 3: Low Freq (k+1 to end)
        comp3 = np.sum(data_npy[:, :, k+1:], axis=-1) if k+1 < N_IMFS else np.zeros((T, C))
        
        # Stack -> [T, C, 3]
        data_processed = np.stack([comp1, comp2, comp3], axis=-1)

        # 5. 标准化 (Scaling)
        if self.scale:
            # 加载 Train 数据计算 Mean/Std
            train_npy_path = os.path.join(self.root_path, f"{base_name}_train_sl{self.seq_len}_cd.npy")
            
            # 读取 Train 并筛选特征
            train_npy = np.load(train_npy_path) # [T_train, Total_C, K]
            train_npy = train_npy[:, target_indices, :] # [T_train, Selected_C, K]
            
            # 还原为原始信号值来 Fit Scaler
            train_sum = np.sum(train_npy, axis=-1) # [T_train, C]
            self.scaler.fit(train_sum)
            
            # 获取 scaler 的参数，形状适配 [1, C, 1] 以便广播
            # mean: [C], scale: [C]
            mean = self.scaler.mean_.reshape(1, C, 1)
            scale = self.scaler.scale_.reshape(1, C, 1)
            
            # --- 关键 Scaling 逻辑 ---
            # data_processed: [T, C, 3]
            # 0: High, 1: Mid, 2: Low
            
            # 1. High Freq & Mid Freq: 仅除以 scale (假设它们是围绕0波动的)
            data_processed[:, :, 0:2] = data_processed[:, :, 0:2] / scale
            
            # 2. Low Freq (Trend): 减去 Mean 并除以 scale (它承担了基准偏移)
            data_processed[:, :, 2:3] = (data_processed[:, :, 2:3] - mean) / scale

        # 6. 转置为模型需要的格式
        # 通常 Time-Series-Library 是 [T, C]
        # 这里为了保持 __getitem__ 方便，我们先存为 [3, T, C]
        self.data_x = data_processed.transpose(2, 0, 1) # [3, T, C]
        self.data_y = data_processed.transpose(2, 0, 1) # [3, T, C]
        self.data_stamp = data_stamp

        # Augmentation (仅对 Train 有效)
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            # 注意：augmentation 库通常期望输入是 [T, C]，这里是 [3, T, C]
            # 直接调用可能会报错，取决于 utils.augmentation 的实现。
            # 这里建议先忽略，或者需要对每一层分别做 augmentation
            pass 