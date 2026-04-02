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

class Dataset_Custom_Decomposed(Dataset):
    def __init__(self, 
                 args, 
                 root_path, 
                 flag='train', 
                 size=None, 
                 features='S', 
                 data_path='ETTh1.csv',
                 target='OT', 
                 scale=True, 
                 time_enc=0, 
                 freq='h', 
                 seasonal_patterns=None,
                 data_format='custom'):
        self.args = args
        self.mnn = args.mnn
        self.data_format = data_format
        # size [seq_len, label_len, pred_len]
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
        self.base_name = os.path.splitext(self.data_path)[0] # 构造文件名
        # k parameter for component merging
        self.k = getattr(args, 'pivot', 1) # 增加默认值防止报错
        self.use_mnn = True if getattr(args, 'use_mnn', 0) == 1 else False
        self.__read_data__()

    def _read_solar_raw(self, local_fp):
        if self.data_path.endswith('.txt'):
            out_fp = os.path.join(self.root_path, 'solar_AL.csv')
            df_raw = []
            with open(local_fp, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    if not line:
                        continue
                    data_line = np.stack([float(i) for i in line.split(',')])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_out = pd.DataFrame(df_raw)
            df_out.to_csv(out_fp, index=False, encoding='utf-8')
        elif self.data_path.endswith('.csv'):
            df_out = pd.read_csv(local_fp)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
        return df_out
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        # decomposition.py 生成的文件名包含 _scaled_cd
        npy_path = os.path.join(self.root_path, f"{self.base_name}_{current_flag_name}_sl{self.seq_len}_scaled_cd.npy")
        
        if os.path.exists(npy_path):
            data_npy = np.load(npy_path) # [Split_Len, Total_Channels, N_IMFS]
        else:
            raise FileNotFoundError(f"Decomposed data not found: {npy_path}")
        
        # 2. 读取原始数据，Solar 与 Custom 保持和 data_loader 一致的读取方式
        file_path = os.path.join(self.root_path, self.data_path)
        is_solar_format = self.data_format == 'solar' or os.path.splitext(self.data_path)[1].lower() == '.txt'
        if is_solar_format:
            df_raw = self._read_solar_raw(file_path)
        else:
            df_raw = pd.read_csv(file_path)

        num_train = int(len(df_raw) * 0.7) 
        num_test = int(len(df_raw)  * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        self.border_map = {
            'ETTh': {
                'start': [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len],
                'end':   [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            },
            'ETTm': {
                'start': [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len],
                'end':   [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            },
            'custom': {
                'start': [0,         num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
                'end':   [num_train, num_train + num_vali,     len(df_raw)]
            }
        }
        
        if 'ETTm' in self.data_path:
            self.borders = self.border_map['ETTm']
        elif 'ETTh' in self.data_path:
            self.borders = self.border_map['ETTh']
        else:
            self.borders = self.border_map['custom']
        
        if is_solar_format:
            if self.features == 'M' or self.features == 'MS':
                df_data = df_raw
            elif self.features == 'S':
                target_idx = int(self.target)
                df_data = df_raw.iloc[:, [target_idx]]
                data_npy = data_npy[:, target_idx:target_idx+1, :]
            else:
                raise ValueError(f"Features {self.features} is not valid")

        else:
            if self.features == 'M' or self.features == 'MS':
                # 取所有数据列
                cols_data = df_raw.columns[1:] 
                df_data = df_raw[cols_data]

            elif self.features == 'S':
                if self.target not in df_raw.columns:
                    raise ValueError(f"Target {self.target} not found.")
                # 找到 target 在 "数据列" (去除date后) 中的索引
                data_cols = list(df_raw.columns[1:])
                target_idx = data_cols.index(self.target)
                df_data = df_raw[[self.target]]
                # NPY 数据也必须只保留 target 对应的通道
                # data_npy: [T, Total_C, K] -> [T, 1, K]
                data_npy = data_npy[:, target_idx:target_idx+1, :]
            else:
                raise ValueError(f"Features {self.features} is not valid")

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

        if is_solar_format:
            data_stamp = np.zeros((end_idx - start_idx, 1))
        else:
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

        #########################################################
        # 4. 合并分量
        #########################################################
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

        self.data_decomp = data_processed                  # 存储分解信号 [T, C, K]
        self.data_original = raw_scaled[start_idx:end_idx] # 存储原始信号 [T, C]
        self.data_stamp = data_stamp
        
        if self.use_mnn and self.set_type == 2:
            self.__read_mnn_data__(self.data_original[start_idx:end_idx])
        
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
        if self.use_mnn and self.set_type == 2:
            seq_x_decomp = self.data_mnn_test[s_begin:s_end, :, :]
           
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
    
    def __read_mnn_data__(self, test_raw_data):
        """
        Read MNN data for test.
        Input: test_raw_data [T, C]
        Output: self.data_mnn_test [T, C, K]
        """
        suffix = "_smoothed"
        prefix = "all" if self.k is None else "pred"
        data_mnn_test_path = os.path.join(self.root_path, f"{prefix}_{self.base_name}_test_sl{self.seq_len}_{self.mnn}_scaled_cd{suffix}.npy")
        data_mnn_test = np.load(data_mnn_test_path)
        num_vars = test_raw_data.shape[-1]
        print(f">>>>>>>>>>>>> data_mnn_test.shape: {data_mnn_test.shape}, test_raw_data.shape: {test_raw_data.shape}")
        
        if data_mnn_test.shape[-1] == self.args.num_imf - 1: # we only learn residual in training mnn
            data_mnn_test = data_mnn_test.reshape(-1, num_vars, self.args.num_imf - 1)
            remain = test_raw_data.values - data_mnn_test.sum(axis=-1)
            self.data_mnn_test = np.concatenate([remain.reshape(-1, num_vars, 1), data_mnn_test], axis=-1) # [T, C, K]
        
        elif data_mnn_test.shape[-1] == self.args.num_imf:
            self.data_mnn_test = data_mnn_test # Abalation: we don't need to learn residual in testing mnn
        
        else:
            raise ValueError(f"data_mnn_test.shape: {data_mnn_test.shape} is not valid")

    def __len__(self):
        # 原 self.data_x 已更名为 self.data_decomp
        return len(self.data_decomp) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_PEMS_Decomposed(Dataset):
    def __init__(
        self, args, root_path, flag='train', size=None, features='M', data_path='PEMS03.npz',
        target=10, scale=True, time_enc=0, freq='h', seasonal_patterns=None):
        self.args = args
        self.mnn = args.mnn
        self.use_residual = True
        self.k = getattr(args, 'pivot', 1) # 增加默认值防止报错
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
        self.base_name = os.path.splitext(self.data_path)[0]
        self.use_mnn = True if getattr(args, 'use_mnn', 0) == 1 else False
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path.replace('.npz', '_scaled.npy'))
        fname_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag_name = fname_map[self.set_type]
        # 加载带 seq_len 的文件名
        npy_path = os.path.join(self.root_path, f"{self.base_name}_{current_flag_name}_sl{self.seq_len}_scaled_cd.npy")
        
        if os.path.exists(npy_path):
            decomp_npy = np.load(npy_path)
            print(f"Loaded decomposed data from {npy_path}")
        else:
            raise FileNotFoundError(f"Decomposed data not found. Looked for {npy_path}.")

        raw_scaled = np.load(data_file)
        # 1. 划分数据集索引
        train_ratio = 0.6
        valid_ratio = 0.2
        len_data = len(raw_scaled)
        len_train = int(train_ratio * len(raw_scaled))
        val_end = int((train_ratio + valid_ratio) * len_data)
        type_len = {0: len_train, 1: val_end - len_train, 2: len_data - val_end}
        s0, e0 = 0, len_train
        s1, e1 = len_train, val_end
        s2, e2 = val_end, len_data
        self.borders = {
            'start': [s0, s1, s2],
            'end': [e0, e1, e2]
        }
        if len(decomp_npy) != type_len[self.set_type]:
            # 这一步非常关键，如果 decomposition.py 的切分逻辑和这里的切分逻辑不一致，这里会报错
            print(f"Error: NPY len ({len(decomp_npy)}) != CSV split len ({type_len[self.set_type]}).")
            exit(0)
        
        if self.features == 'S':
            decomp_npy = decomp_npy[:, [self.target], :]    # (T, 1, K)
            raw_scaled = raw_scaled[:, [self.target]]       # (T, 1)
        
        # 读取原始数据并标准化
        # if self.scale:
        #     self.scaler.fit(raw_scaled[s0:e0])
        #     raw_scaled = self.scaler.transform(raw_scaled)            
        # else:
        #     raw_scaled = raw_scaled
        # np.save(f"{self.base_name}_scaled_dataset.npy", raw_scaled)
        print(f">>>>>>>>>>>>> Scale, raw_data.shape: {raw_scaled.shape}, raw_scaled.shape: {raw_scaled.shape}")
        # 读取分量并合并为3个分量
        data_processed = merge_components(decomp_npy, self.k) # [T, C, K]
        print(f">>>>>>>>>>>>> {current_flag_name}, Use MNN: {self.use_mnn}, data_processed.shape: {data_processed.shape}")
        if self.use_mnn and self.set_type == 2:
            self.__read_mnn_data__(raw_scaled[s2:e2])
            
        start_idx = self.borders['start'][self.set_type]
        end_idx = self.borders['end'][self.set_type]
        self.data_decomp = data_processed                  # 存储分解信号 [T, C, K]
        self.data_original = raw_scaled[start_idx:end_idx] # 存储原始信号 [T, C]
        
        # 验证 self.data_decomp 和 self.data_original 是否相等
        # print(f">>>>>>>>>>>>> self.data_decomp.shape: {self.data_decomp.shape}, self.data_original.shape: {self.data_original.shape}")  
        # valid_decomp = data_processed.sum(axis=-1)
        # print(f">>>>>>>>>>>>> max(abs(valid_decomp)): {np.max(np.abs(valid_decomp))}")
        # print(f">>>>>>>>>>>>> max(abs(self.data_original)): {np.max(np.abs(self.data_original))}")
        # if np.allclose(valid_decomp, self.data_original):
        #     print(f">>>>>>>>>>>>> valid_decomp == self.data_original: True")
        # else:
        #     print(f">>>>>>>>>>>>> valid_decomp == self.data_original: False")
        #     exit(0)
        # # ================== 诊断代码结束 ==================
        
        
    def __read_mnn_data__(self, test_raw_data):
        k = None if self.args.num_imf == 15 else self.k
        suffix = "_smoothed"
        prefix = "all" if k is None else "pred"
        data_mnn_test_path = os.path.join(self.root_path, f"{prefix}_{self.base_name}_test_sl{self.seq_len}_{self.mnn}_scaled_cd{suffix}.npy")
        # =======================================================
        data_mnn_test = np.load(data_mnn_test_path)
        print(f">>>>>>>>>>>>> Loaded MNN test data from {data_mnn_test_path}, shape: {data_mnn_test.shape}")
        C = test_raw_data.shape[-1]
        if data_mnn_test.shape[-1] == self.args.num_imf - 1: # Use Residual Data for Test
            data_mnn_test = data_mnn_test.reshape(-1, C, self.args.num_imf - 1) # (T, C, 2)
            remain = test_raw_data - data_mnn_test.sum(axis=-1) # (T, C)
            self.data_mnn_test = np.concatenate([remain.reshape(-1, C, 1), data_mnn_test], axis=-1)
            print(f">>>>>>>>>>>>> After Residual, data_mnn_test.shape: {data_mnn_test.shape}")
            
        elif data_mnn_test.shape[-1] == self.args.num_imf:
            self.data_mnn_test = data_mnn_test
        else:
            raise ValueError(f"data_mnn_test.shape: {data_mnn_test.shape} is not valid")
                
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # ====================================================
        # 1. 构建输入 seq_x: [Seq_Len, C, K+1]
        # ====================================================
        # A. 获取分解部分 [Seq_Len, C, K]
        seq_x_decomp = self.data_decomp[s_begin:s_end, :, :]
        
        # B. 获取原始部分 [Seq_Len, C] -> 扩展为 [Seq_Len, C, 1]
        seq_x_original = self.data_original[s_begin:s_end, :]
        seq_x_original = seq_x_original[:, :, np.newaxis] 

        # C. 拼接: 原始信号在第0位 [Seq_Len, C, 1+K]
        if self.use_mnn and self.set_type == 2:
            seq_x_decomp = self.data_mnn_test[s_begin:s_end, :, :]
           
        seq_x = np.concatenate([seq_x_original, seq_x_decomp], axis=-1)
        
        # ====================================================
        # 2. 构建标签 seq_y: [Label_Len + Pred_Len, C, K+1]
        # ====================================================
        # A. 获取分解标签
        seq_y_decomp = self.data_decomp[r_begin:r_end, :, :]
        
        # B. 获取原始标签
        seq_y_original = self.data_original[r_begin:r_end, :]
        seq_y_original = seq_y_original[:, :, np.newaxis]
        
        # C. 拼接
        seq_y = np.concatenate([seq_y_original, seq_y_decomp], axis=-1)
        
        # ====================================================
        # 3. 时间戳 (不变)
        # ====================================================
        seq_x_mark = torch.zeros((seq_x.shape[0], seq_x.shape[1], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], seq_y.shape[1], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 2:
            return (self.data_decomp.shape[0] - self.seq_len - self.pred_len + 1) // 12
        else:
            return self.data_decomp.shape[0] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # data shape: [Batch, Seq_Len, Nodes]
        shape = data.shape
        # 展平为 [Batch * Seq_Len, Nodes] 进行反归一化
        data_flat = data.reshape(-1, shape[-1])
        data_inv = self.scaler.inverse_transform(data_flat)
        # 还原形状
        return data_inv.reshape(shape)
    
    