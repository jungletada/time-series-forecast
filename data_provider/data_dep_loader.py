import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
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


def split_residual(data_npy, data_processed):
    """Ablation Split residual signal from the processed data"""
    T, C, N_IMFS = data_npy.shape
    print(f"data_processed.shape: {data_processed.shape}")
    print(f"data_npy.shape: {data_npy.shape}")
    raw_signal = data_npy.sum(axis=-1)
    print(f"raw_signal.shape: {raw_signal.shape}")
    high_freq = raw_signal - data_processed[:, :, 1] - data_processed[:, :, 2]
    data_processed[:, :, 0] = high_freq
    return data_processed

class Dataset_ETT_Decomposed(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, time_enc=0, freq='h', seasonal_patterns=None,):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.mnn = args.mnn
        self.use_residual = True
        use_mnn = getattr(args, 'use_mnn', 0)

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
        # 根据实际 decomposition.py 的 max_imfs (例如10) 来调整这个 k
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
        
        # 1. 构造文件名 (确保 decomposition.py 生成的文件名包含 _cd)
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        npy_path = os.path.join(self.root_path, f"{base_name}_{current_flag_name}_sl{self.seq_len}_cd.npy")
        pred_train_npy_path = os.path.join(self.root_path, f"pred_{base_name}_train_sl{self.seq_len}_cd.npy")
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path)
        else:
            raise FileNotFoundError(f"Decomposed data not found. Looked for {npy_path}")
        
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
        s1, e1 = self.borders['start'][1], self.borders['end'][1]
        s2, e2 = self.borders['start'][2], self.borders['end'][2]
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
        T, C, N_IMFS = data_npy.shape
        if self.use_mnn:
            if self.set_type == 0:
                mnn_train_npy_path = os.path.join(self.root_path, f"pred_{base_name}_train_sl{self.seq_len}_{self.mnn}_cd.npy")
                if os.path.exists(mnn_train_npy_path):
                    data_mnn_npy_train = np.load(mnn_train_npy_path)
                    print(f"Loaded MNN data from {mnn_train_npy_path}, shape: {data_mnn_npy_train.shape}")
                else:
                    raise FileNotFoundError(f"MNN data not found. Looked for {mnn_train_npy_path}.")
                print(f">>>>>>>>>>>>> Using MNN data for training")
                if self.use_residual:
                    train_raw_data = df_data[s0:e0]
                    data_mnn_npy_train = data_mnn_npy_train.reshape(-1, 1, self.args.num_imf - 1)
                    high = train_raw_data.values - data_mnn_npy_train.sum(axis=-1)
                    data_mnn_npy_train = np.concatenate([high.reshape(-1, 1, 1), data_mnn_npy_train], axis=-1)
                    data_processed = data_mnn_npy_train
                else:
                    data_processed = data_mnn_npy_train

            elif self.set_type == 1:
                # print(f">>>>>>>>>>>>> Using MNN data for validation")
                data_processed = merge_components(data_npy, self.k)

            if self.set_type == 2: # load mnn data for test
                k = None if self.args.num_imf == 10 else self.k
                suffix = "_smoothed"
                prefix = "all" if k is None else "pred"
                data_mnn_test_path = os.path.join(self.root_path, f"{prefix}_{base_name}_test_sl{self.seq_len}_{self.mnn}_cd{suffix}.npy")
                # ================== Use Residual Data for Test=====================================
                data_mnn_test = np.load(data_mnn_test_path)
                print(f">>>>>>>>>>>>> data_mnn_test.shape: {data_mnn_test.shape}")
                if self.use_residual:
                    s2, e2 = self.borders['start'][2], self.borders['end'][2]
                    test_raw_data = df_data[s2:e2]
                    if data_mnn_test.shape[-1] == self.args.num_imf - 1:
                        data_mnn_test = data_mnn_test.reshape(-1, 1, self.args.num_imf - 1)
                        high = test_raw_data.values - data_mnn_test.sum(axis=-1)
                        data_mnn_test = np.concatenate([high.reshape(-1, 1, 1), data_mnn_test], axis=-1)
                    elif data_mnn_test.shape[-1] == self.args.num_imf:
                        data_mnn_test = data_mnn_test.reshape(-1, 1, self.args.num_imf)
                        high = test_raw_data.values - data_mnn_test.sum(axis=-1)
                        data_mnn_test = np.concatenate([high.reshape(-1, 1, 1), data_mnn_test[:, :, 1:]], axis=-1)
                    else:
                        raise ValueError(f"data_mnn_test.shape: {data_mnn_test.shape} is not valid")
                # ================== Use Residual Data for Test=====================================
                print(f">>>>>>>>>>>>> After Residual, data_mnn_test.shape: {data_mnn_test.shape}")
                data_processed = merge_components(data_npy, self.k)

        else:
            data_processed = merge_components(data_npy, self.k)
        
        print(f">>>>>>>>>>>>> {current_flag_name} {self.use_mnn}: data_processed.shape: {data_processed.shape}")
            
        # 5. 标准化 (Scaling)
        if self.scale:
            # 加载 Train 数据计算 Mean/Std
            train_raw_data = df_data[s0:e0]
            self.scaler.fit(train_raw_data.values)
            # 获取 scaler 的参数，形状适配 [1, C, 1] 以便广播
            # mean: [C], scale: [C]
            mean = self.scaler.mean_.reshape(1, C, 1)
            scale = self.scaler.scale_.reshape(1, C, 1)
            # 对于前面的分量 (High/Mid Freq): 仅缩放 (Scale)，不平移
            # 0 到 K-2 是高频/中频
            data_processed[:, :, :-1] = data_processed[:, :, :-1] / scale
            # 对于最后一个分量 (Trend/Low Freq): 缩放 + 平移 (减去均值)
            # 这样 sum(components) = (H/s + ... + (L-m)/s) = (Sum-m)/s = (X-m)/s
            data_processed[:, :, -1:] = (data_processed[:, :, -1:] - mean) / scale
          
            # 对于 MNN 测试数据的处理同理
            if self.set_type == 2 and self.use_mnn:
                data_mnn_test[:, :, :-1] = data_mnn_test[:, :, :-1] / scale
                data_mnn_test[:, :, -1:] = (data_mnn_test[:, :, -1:] - mean) / scale

        # 6. 转置为模型需要的格式
        # 通常 Time-Series-Library 是 [T, C]
        if self.set_type == 2 and self.use_mnn:
            self.data_x = data_mnn_test.transpose(2, 0, 1) # [K, T, C]
        else:
            self.data_x = data_processed.transpose(2, 0, 1) # [K, T, C]
        self.data_y = data_processed.transpose(2, 0, 1) # [K, T, C]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # self.data_x: [K, T, C]
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
        data: [Batch, K, T, C] 或者 [Batch, T, C] (如果模型已经求和了)
        """
        # 如果输入是分开的 3 个分量，先求和
        if data.ndim == 4:
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
        target='OT', scale=True, time_enc=0, freq='h', seasonal_patterns=None,):
        # [seq_len, label_len, pred_len]
        self.args = args
        self.mnn = args.mnn
        use_mnn = getattr(args, 'use_mnn', 0)
        self.use_mnn = True if use_mnn == 1 else False
        self.k = self.args.selected_k
        self.use_residual = True
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # initialize
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.target_idx = None
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
        print(f">>>>>>>>>>>>> base_name: {base_name}, cfg_name: {cfg_name}")
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        # 加载带 seq_len 的文件名
        npy_path = os.path.join(self.root_path, f"{base_name}_{current_flag_name}_sl{self.seq_len}_cd.npy")
        
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path)
            print(f"Loaded decomposed data from {npy_path}, shape: {data_npy.shape}")
        else:
            raise FileNotFoundError(f"Decomposed data not found. Looked for {npy_path}.")

        # 2. 读取 CSV 以获取特征索引和时间戳
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)
        num_train = int(len(df_raw) * 0.7) 
        num_test = int(len(df_raw)  * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        self.borders = {
            'start': [0,         num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            'end':   [num_train, num_train + num_vali, len(df_raw)]
        }
        # --- Feature Selection (S or M) ---
        if self.features == 'M' or self.features == 'MS': # 排除第一列 date，取剩余所有
            cols_data = df_raw.columns[1:] # 对应的 Numpy 索引就是 0 到 C-1
            df_data = df_raw[cols_data]
            target_indices = list(range(len(cols_data))) 
        elif self.features == 'S': # 只取 target 列
            if self.target not in df_raw.columns:
                 raise ValueError(f"Target {self.target} not found in CSV columns.")
            # 找到 target 在 "数据列" (去除date后) 中的索引
            df_data = df_raw[[self.target]]
            data_cols = list(df_raw.columns[1:])
            target_idx = data_cols.index(self.target)
            target_indices = [target_idx]
            self.target_idx = target_idx
       
        # data_npy: [T, Total_Channels, K] -> [T, Selected_Channels, K]
        data_npy = data_npy[:, target_indices, :]
        start_idx = self.borders['start'][self.set_type]
        end_idx = self.borders['end'][self.set_type]
        s0, e0 = self.borders['start'][0], self.borders['end'][0]
        s1, e1 = self.borders['start'][1], self.borders['end'][1]
        s2, e2 = self.borders['start'][2], self.borders['end'][2]
        # 严格对齐检查
        if len(data_npy) != (end_idx - start_idx):
            print(f"!!!!!!!!!!!  Error: NPY len ({len(data_npy)}) != CSV split len ({end_idx - start_idx}).")
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

        # 4. 读取分量并且合并为3个分量 (Merge Components)
        T, C, N_IMFS = data_npy.shape

        if self.use_mnn:
            if self.set_type == 0:
                mnn_train_npy_path = os.path.join(self.root_path, f"pred_{base_name}_train_sl{self.seq_len}_{self.mnn}_cd.npy")
                if os.path.exists(mnn_train_npy_path):
                    data_mnn_npy_train = np.load(mnn_train_npy_path)
                    print(f"Loaded MNN data from {mnn_train_npy_path}, shape: {data_mnn_npy_train.shape}")
                else:
                    raise FileNotFoundError(f"MNN data not found. Looked for {mnn_train_npy_path}.")
                print(f">>>>>>>>>>>>> Using MNN data for training")
                if self.use_residual:
                    train_raw_data = df_data[s0:e0]
                    data_mnn_npy_train = data_mnn_npy_train.reshape(-1, 1, self.args.num_imf - 1)
                    high = train_raw_data.values - data_mnn_npy_train.sum(axis=-1)
                    data_mnn_npy_train = np.concatenate([high.reshape(-1, 1, 1), data_mnn_npy_train], axis=-1)
                    data_processed = data_mnn_npy_train
                else:
                    data_processed = data_mnn_npy_train

            elif self.set_type == 1:
                # print(f">>>>>>>>>>>>> Using MNN data for validation")
                data_processed = merge_components(data_npy, self.k)

            if self.set_type == 2: # load mnn data for test
                k = None if self.args.num_imf == 10 else self.k
                suffix = "_smoothed"
                prefix = "all" if k is None else "pred"
                data_mnn_test_path = os.path.join(self.root_path, f"{prefix}_{base_name}_test_sl{self.seq_len}_{self.mnn}_cd{suffix}.npy")
                # ================== Use Residual Data for Test=====================================
                data_mnn_test = np.load(data_mnn_test_path)
                print(f">>>>>>>>>>>>> data_mnn_test.shape: {data_mnn_test.shape}")
                if self.use_residual:
                    s2, e2 = self.borders['start'][2], self.borders['end'][2]
                    test_raw_data = df_data[s2:e2]
                    if data_mnn_test.shape[-1] == self.args.num_imf - 1:
                        data_mnn_test = data_mnn_test.reshape(-1, 1, self.args.num_imf - 1)
                        high = test_raw_data.values - data_mnn_test.sum(axis=-1)
                        data_mnn_test = np.concatenate([high.reshape(-1, 1, 1), data_mnn_test], axis=-1)
                    elif data_mnn_test.shape[-1] == self.args.num_imf:
                        data_mnn_test = data_mnn_test.reshape(-1, 1, self.args.num_imf)
                        high = test_raw_data.values - data_mnn_test.sum(axis=-1)
                        data_mnn_test = np.concatenate([high.reshape(-1, 1, 1), data_mnn_test[:, :, 1:]], axis=-1)
                    else:
                        raise ValueError(f"data_mnn_test.shape: {data_mnn_test.shape} is not valid")
                # ================== Use Residual Data for Test=====================================
                print(f">>>>>>>>>>>>> After Residual, data_mnn_test.shape: {data_mnn_test.shape}")
                data_processed = merge_components(data_npy, self.k)

        else:
            data_processed = merge_components(data_npy, self.k)
        
        print(f">>>>>>>>>>>>> {current_flag_name} {self.use_mnn}: data_processed.shape: {data_processed.shape}")

        # 5. 标准化 (Scaling)
        if self.scale:
            # 加载 Train 数据计算 Mean/Std
            train_raw_data = df_data[s0:e0]
            self.scaler.fit(train_raw_data.values)
            # =======================================================
            # # 还原为原始信号值来 Fit Scaler
            # train_npy_path = os.path.join(self.root_path, f"{base_name}_train_sl{self.seq_len}_cd.npy")
            # train_npy = np.load(train_npy_path)         # [T_train, Total_C, K]
            # train_npy = train_npy[:, target_indices, :] # [T_train, Selected_C, K]
            # train_sum = np.sum(train_npy, axis=-1)      # [T_train, C]
            # # 判断train_sum和train_raw在数值上是否相等
            # if np.allclose(train_sum, train_raw_data.values):
            #     print("train_sum 和 train_raw 数值相等")
            # else:
            #     print("train_sum 和 train_raw 数值不相等")
            # # =======================================================
            # 获取 scaler 的参数，形状适配 [1, C, 1] 以便广播
            # mean: [C], scale: [C]
            mean = self.scaler.mean_.reshape(1, C, 1)
            scale = self.scaler.scale_.reshape(1, C, 1)
            # 对于前面的分量 (High/Mid Freq): 仅缩放 (Scale)，不平移
            # 0 到 K-2 是高频/中频
            data_processed[:, :, :-1] = data_processed[:, :, :-1] / scale
            # 对于最后一个分量 (Trend/Low Freq): 缩放 + 平移 (减去均值)
            # 这样 sum(components) = (H/s + ... + (L-m)/s) = (Sum-m)/s = (X-m)/s
            data_processed[:, :, -1:] = (data_processed[:, :, -1:] - mean) / scale
          
            # 对于 MNN 测试数据的处理同理
            if self.set_type == 2 and self.use_mnn:
                data_mnn_test[:, :, :-1] = data_mnn_test[:, :, :-1] / scale
                data_mnn_test[:, :, -1:] = (data_mnn_test[:, :, -1:] - mean) / scale

        # 6. 转置为模型需要的格式
        # 通常 Time-Series-Library 是 [T, C]
        if self.set_type == 2 and self.use_mnn:
            self.data_x = data_mnn_test.transpose(2, 0, 1) # [K, T, C]
        else:
            self.data_x = data_processed.transpose(2, 0, 1) # [K, T, C]
        self.data_y = data_processed.transpose(2, 0, 1) # [K, T, C]
        self.data_stamp = data_stamp
  
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # self.data_x: [K, T, C]
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
        data: [Batch, K, T, C] 或者 [Batch, T, C]
        """
        # 1. 聚合分量 (如果输入是分离的IMFs)
        if data.ndim == 4: 
            data = data.sum(dim=1) # [Batch, T, C]
            
        # 2. 转为 Numpy
        if hasattr(data, 'cpu'): data = data.cpu().numpy()
        
        shape = data.shape
        # 3. Flatten time dims
        if data.ndim == 3:
            data = data.reshape(-1, shape[-1])
        inverse_data = self.scaler.inverse_transform(data) 
        
        return inverse_data.reshape(shape)

class Dataset_PEMS_Decomposed(Dataset):
    def __init__(
        self, args, root_path, flag='train', size=None, features='S', data_path='PEMS03.npz',
        target=10, scale=True, time_enc=0, freq='h', seasonal_patterns=None):
        self.args = args
        self.mnn = args.mnn
        use_mnn = getattr(args, 'use_mnn', 0)
        self.use_residual = True
        self.use_mnn = True if use_mnn == 1 else False
        self.k = 4 # self.args.selected_k
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = int(target) 
        self.scale = scale
        self.time_enc = time_enc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_name = os.path.splitext(self.data_path)[0]
        data_file = os.path.join(self.root_path, self.data_path)
        
         # 1. 构造文件名 (确保 decomposition.py 生成的文件名包含 _cd)
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        # 加载带 seq_len 的文件名
        npy_path = os.path.join(self.root_path, f"{base_name}_{current_flag_name}_sl{self.seq_len}_cd.npy")
        
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path)
            print(f"Loaded decomposed data from {npy_path}")
        else:
            raise FileNotFoundError(f"Decomposed data not found. Looked for {npy_path}.")

        raw_data = np.load(data_file, allow_pickle=True)
        raw_data = raw_data['data'][:, :, 0] # Use Flow
        
        # 1. 划分数据集索引
        train_ratio = 0.6
        valid_ratio = 0.2
        len_data = len(raw_data)
        len_train = int(train_ratio * len(raw_data))
        val_end = int((train_ratio + valid_ratio) * len_data)
        type_len = {0: len_train, 1: val_end - len_train, 2: len_data - val_end}
        s0, e0 = 0, len_train
        s1, e1 = len_train, val_end
        s2, e2 = val_end, len_data

        if len(data_npy) != type_len[self.set_type]:
            # 这一步非常关键，如果 decomposition.py 的切分逻辑和这里的切分逻辑不一致，这里会报错
            print(f"Error: NPY len ({len(data_npy)}) != CSV split len ({type_len[self.set_type]}).")
            exit(0)
                
        if self.features == 'S':
            data_npy = data_npy[:, [self.target], :]
            raw_data = raw_data[:, [self.target]]
        
        # 4. 读取分量并且合并为3个分量 (Merge Components)
        T, C = raw_data.shape
        if self.use_mnn:
            if self.set_type == 0:
                mnn_train_npy_path = os.path.join(self.root_path, f"pred_{base_name}_train_sl{self.seq_len}_{self.mnn}_cd.npy")
                if os.path.exists(mnn_train_npy_path):
                    data_mnn_npy_train = np.load(mnn_train_npy_path)
                    print(f"Loaded MNN data from {mnn_train_npy_path}, shape: {data_mnn_npy_train.shape}")
                else:
                    raise FileNotFoundError(f"MNN data not found. Looked for {mnn_train_npy_path}.")
                print(f">>>>>>>>>>>>> Using MNN data for training")
                if self.use_residual:
                    train_raw_data = raw_data[s0:e0]
                    data_mnn_npy_train = data_mnn_npy_train.reshape(-1, 1, self.args.num_imf - 1)
                    high = train_raw_data - data_mnn_npy_train.sum(axis=-1)
                    data_mnn_npy_train = np.concatenate([high.reshape(-1, 1, 1), data_mnn_npy_train], axis=-1)
                    data_processed = data_mnn_npy_train
                else:
                    data_processed = data_mnn_npy_train

            elif self.set_type == 1:
                # print(f">>>>>>>>>>>>> Using MNN data for validation")
                data_processed = merge_components(data_npy, self.k)

            if self.set_type == 2: # load mnn data for test
                data_processed = merge_components(data_npy, self.k)
                k = None if self.args.num_imf == 10 else self.k
                suffix = "_smoothed"
                prefix = "all" if k is None else "pred"
                data_mnn_test_path = os.path.join(self.root_path, f"pred_{base_name}_test_sl{self.seq_len}_{self.mnn}_cd{suffix}.npy")
                # ================== Use Residual Data for Test=====================================
                data_mnn_test = np.load(data_mnn_test_path)
                print(f">>>>>>>>>>>>> Loaded MNN test data from {data_mnn_test_path}, shape: {data_mnn_test.shape}")
                if self.use_residual:
                    test_raw_data = raw_data[s2:e2]
                    if data_mnn_test.shape[-1] == self.args.num_imf - 1:
                        data_mnn_test = data_mnn_test.reshape(-1, 1, self.args.num_imf - 1) # (T, 1, 2)
                        high = test_raw_data - data_mnn_test.sum(axis=-1) # (T, 1, 1)
                        data_mnn_test = np.concatenate([high.reshape(-1, 1, 1), data_mnn_test], axis=-1)
                        # data_mnn_test = data_processed
                        print(f">>>>>>>>>>>>> After Residual, data_mnn_test.shape: {data_mnn_test.shape}")
                    elif data_mnn_test.shape[-1] == self.args.num_imf:
                        data_mnn_test = data_mnn_test.reshape(-1, 1, self.args.num_imf)
                        high = test_raw_data - data_mnn_test.sum(axis=-1)
                        data_mnn_test = np.concatenate([high.reshape(-1, 1, 1), data_mnn_test[:, :, 1:]], axis=-1)
                    else:
                        raise ValueError(f"data_mnn_test.shape: {data_mnn_test.shape} is not valid")
                # ================== Use Residual Data for Test=====================================
                # print(f">>>>>>>>>>>>> After Residual, data_mnn_test.shape: {data_mnn_test.shape}")
        else:
            data_processed = merge_components(data_npy, self.k)
        
        print(f">>>>>>>>>>>>> {current_flag_name} {self.use_mnn}: data_processed.shape: {data_processed.shape}")

        # 5. 标准化 (Scaling)
        if self.scale:
            # 加载 Train 数据计算 Mean/Std
            train_raw_data = raw_data[s0:e0]
            self.scaler.fit(train_raw_data)
            # # =======================================================
            # # 还原为原始信号值来 Fit Scaler
            # train_npy_path = os.path.join(self.root_path, f"{base_name}_train_sl{self.seq_len}_cd.npy")
            # train_npy = np.load(train_npy_path)         # [T_train, Total_C, K]
            # train_npy = train_npy[:, [self.target], :] # [T_train, Selected_C, K]
            # train_sum = np.sum(train_npy, axis=-1)      # [T_train, C]
            # # 判断train_sum和train_raw在数值上是否相等
            # if np.allclose(train_sum, train_raw_data):
            #     print("train_sum 和 train_raw 数值相等")
            # else:
            #     print("train_sum 和 train_raw 数值不相等")
            # # =======================================================
            # 获取 scaler 的参数，形状适配 [1, C, 1] 以便广播
            # mean: [C], scale: [C]
            mean = self.scaler.mean_.reshape(1, C, 1)
            scale = self.scaler.scale_.reshape(1, C, 1)
            # 对于前面的分量 (High/Mid Freq): 仅缩放 (Scale)，不平移
            # 0 到 K-2 是高频/中频
            data_processed[:, :, :-1] = data_processed[:, :, :-1] / scale
            # 对于最后一个分量 (Trend/Low Freq): 缩放 + 平移 (减去均值)
            # 这样 sum(components) = (H/s + ... + (L-m)/s) = (Sum-m)/s = (X-m)/s
            data_processed[:, :, -1:] = (data_processed[:, :, -1:] - mean) / scale
          
            # 对于 MNN 测试数据的处理同理
            if self.set_type == 2 and self.use_mnn:
                data_mnn_test[:, :, :-1] = data_mnn_test[:, :, :-1] / scale
                data_mnn_test[:, :, -1:] = (data_mnn_test[:, :, -1:] - mean) / scale

        # 6. 转置为模型需要的格式
        # 通常 Time-Series-Library 是 [T, C]
        if self.set_type == 2 and self.use_mnn:
            self.data_x = data_mnn_test.transpose(2, 0, 1) # [K, T, C]
        else:
            self.data_x = data_processed.transpose(2, 0, 1) # [K, T, C]
        self.data_y = data_processed.transpose(2, 0, 1) # [K, T, C]
        # self.data_stamp = data_stamp
  
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[:, s_begin:s_end]
        seq_y = self.data_y[:,r_begin:r_end]

        seq_x_mark = torch.zeros((seq_x.shape[0], seq_x.shape[1], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], seq_y.shape[1], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 2:
            return (self.data_x.shape[1] - self.seq_len - self.pred_len + 1) // 12
        else:
            return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_M4_Decomposed(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path=None,
                 target='OT', scale=False, inverse=False, time_enc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.root_path = root_path
        self.flag = flag
        # self.features = features
        # self.target = target
        # self.scale = scale # M4 通常不建议在 Dataset 层做 Global Scaling
        # self.inverse = inverse
        # self.time_enc = time_enc
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # M4 特有参数
        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        
        # 获取 selected_k
        self.k = getattr(self.args, 'selected_k', 2)
        
        self.__read_data__()

    def __read_data__(self):
        # ==========================================
        # Part 1: 加载 IDs (为了修复报错)
        # ==========================================
        dataset_info = M4Dataset.load(training=True, dataset_file=self.root_path)
        
        # 筛选当前频率对应的 IDs
        self.ids = np.array([
            i for i in dataset_info.ids[dataset_info.groups == self.seasonal_patterns]
        ])

        # ==========================================
        # Part 2: 加载分解后的数据 (.npy)
        # ==========================================
        # 1. 构建文件名
        # 始终加载 train_cd.npy，因为它是历史输入源
        file_name = f"M4_{self.seasonal_patterns}_train_cd.npy"
        file_path = os.path.join(self.root_path, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Decomposed M4 data not found: {file_path}")
            
        # 2. 加载 Object Array
        loaded_data = np.load(file_path, allow_pickle=True)
        
        # 安全检查：确保 ID 数量和加载的数据序列数量一致
        if len(self.ids) != len(loaded_data):
            print(f"Warning: IDs count ({len(self.ids)}) != Loaded Data count ({len(loaded_data)})")
            # 通常只要 frequency 对上了，这里是一致的。如果不一致，说明分解文件版本不对。

        self.timeseries = []
        
        # 3. 预处理：Merge Components
        for series in loaded_data:
            T, N_IMFS = series.shape
            # [T, N_IMFS] -> [T, 1, N_IMFS]
            series_expanded = series.reshape(T, 1, N_IMFS)
            # Merge -> [T, 1, 3]
            merged = merge_components(series_expanded, self.k)
            # Transpose -> [3, T, 1]
            merged = merged.transpose(2, 0, 1) 
            
            self.timeseries.append(merged)

    def __getitem__(self, index):
        # 1. 初始化 Data 和 Mask
        # shape: [3, seq_len, 1]
        insample = np.zeros((3, self.seq_len, 1))
        insample_mask = np.zeros((3, self.seq_len, 1)) # Mask 初始化为 0
        
        # shape: [3, label_len + pred_len, 1]
        outsample = np.zeros((3, self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((3, self.pred_len + self.label_len, 1)) # Mask 初始化为 0

        # 获取第 index 条序列: [3, Total_Len, 1]
        sampled_timeseries = self.timeseries[index]
        total_len = sampled_timeseries.shape[1]
        
        # 随机采样切分点
        cut_point = np.random.randint(
            low=max(1, total_len - self.window_sampling_limit),
            high=total_len,
            size=1)[0]

        # --- 构造 Input ---
        # 取出窗口
        insample_window = sampled_timeseries[:, max(0, cut_point - self.seq_len):cut_point, :]
        win_len = insample_window.shape[1]
        
        # 填充数据 (填在末尾)
        insample[:, -win_len:, :] = insample_window
        # 填充 Mask (有效数据部分设为 1)
        insample_mask[:, -win_len:, :] = 1.0

        # --- 构造 Output ---
        # 取出窗口
        outsample_window = sampled_timeseries[:, 
                           max(0, cut_point - self.label_len):min(total_len, cut_point + self.pred_len), :]
        out_win_len = outsample_window.shape[1]
        
        # 填充数据 (填在开头)
        outsample[:, :out_win_len, :] = outsample_window
        # 填充 Mask (有效数据部分设为 1)
        outsample_mask[:, :out_win_len, :] = 1.0
        
        # 返回: x, y, x_mark(即insample_mask), y_mark(即outsample_mask)
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        """
        M4 反归一化
        由于我们没有做 Global Scaling，这里的反归一化 = 将3个分量求和
        data shape: [Batch, 3, T, C]
        """
        # 如果是 [Batch, 3, T, C]，则在 dim=1 求和
        if data.ndim == 4 and data.shape[1] == 3:
            return data.sum(dim=1)
        # 如果已经是 [Batch, T, C]，说明已经求和过了
        return data

    def last_insample_window(self):
        """
        用于测试阶段：获取所有序列最后 seq_len 长度的数据
        Return: [Num_Series, 3, Seq_Len, 1]
        """
        insample = np.zeros((len(self.timeseries), 3, self.seq_len, 1))
        
        for i, ts in enumerate(self.timeseries):
            # ts: [3, Total_Len, 1]
            ts_len = ts.shape[1]
            # 取最后一段
            last_window = ts[:, -self.seq_len:, :]
            win_len = last_window.shape[1]
            
            # 填充
            insample[i, :, -win_len:, :] = last_window
            
        return insample, None # mask省略

