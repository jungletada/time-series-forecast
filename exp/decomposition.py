import os
import numpy as np
import pandas as pd
from PyEMD import EMD
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def merge_components(raw_signal, data_decomposed, k):
    if k is None:
        return data_decomposed
    # 处理分解数据的通道合并
    N_IMFS = data_decomposed.shape[-1]
    k = min(k, N_IMFS - 1)
    if k > 0 and k < N_IMFS - 1: 
        comp1 = np.sum(data_decomposed[:, :, :k], axis=-1)
        comp2 = data_decomposed[:, :, k]
        comp3 = raw_signal - comp1 - comp2
    elif k == 0: # The first component
        comp1 = data_decomposed[:, :, 0]
        comp2 = data_decomposed[:, :, 1]
        comp3 = raw_signal - comp1 - comp2
    elif k == N_IMFS - 1: # The last component
        comp1 = np.sum(data_decomposed[:, :, :N_IMFS - 2], axis=-1)
        comp2 = data_decomposed[:, :, N_IMFS - 2]
        comp3 = raw_signal - comp1 - comp2

    decomp_data = np.stack([comp1, comp2, comp3], axis=-1)
    return decomp_data

class LongTermDecomposition:
    def __init__(self, data_type, root_path, data_file, 
                 selected_k=2, max_imfs=15, seq_len=96, scale=True):
        self.data_type = data_type
        self.root_path = root_path
        self.data_file = data_file
        self.file_path = os.path.join(root_path, data_file)
        self.max_imfs = max_imfs
        self.seq_len = seq_len
        self.selected_k = selected_k
        self.scale = scale  # 是否在分解前归一化
        
        self.emd = EMD()
        self.emd.MAX_ITERATION = 200 
        
        # 初始化 Scaler
        if self.scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None 

        print(f"   Decomposition: {self.file_path}\n"
              f"   max_imfs: {self.max_imfs}\n"
              f"   seq_len: {self.seq_len}\n"
              f"   scale: {self.scale}")

    def _get_borders(self, total_len):
        num_train = int(total_len * 0.7)
        num_test = int(total_len  * 0.2)
        num_val = total_len - num_train - num_test

        if self.data_type == 'ETTh':
            border = {
                'start': [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len],
                'end':   [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            }
        elif self.data_type == 'ETTm':
            border = {
                'start': [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len],
                'end':   [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            }
        elif self.data_type == 'custom':
            # print(f"num_train: {num_train}, num_test: {num_test}, num_val: {num_val}")
            border = {
                'start': [0, num_train - self.seq_len, total_len - num_test - self.seq_len],
                'end':   [num_train, num_train + num_val, total_len]
            }
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")
        return border

    def _decompose_and_pad(self, series_values):
        try:
            # EMD 分解
            imfs = self.emd.emd(series_values).T 
            T, n_imfs = imfs.shape
            # ========================================
            result = np.zeros((T, self.max_imfs))
            if n_imfs >= self.max_imfs: 
                result[:, :self.max_imfs-1] = imfs[:, :self.max_imfs-1]
                result[:, self.max_imfs-1] = np.sum(imfs[:, self.max_imfs-1:], axis=1)  
            else:
                result[:, :n_imfs] = imfs
            return result

        except Exception as e:
            return np.zeros((len(series_values), self.max_imfs))

    def _process_column(self, full_series, border):
        # 注意：full_series 已经是归一化后的数据了（如果 scale=True）
        
        # 1. Train Set
        train_raw = full_series[border['start'][0]:border['end'][0]]
        train_decomp = self._decompose_and_pad(train_raw)
        
        # 2. Validation Set
        val_raw = full_series[0:border['end'][1]] 
        val_decomp_full = self._decompose_and_pad(val_raw)
        val_decomp_cd = val_decomp_full[border['start'][1]:border['end'][1]]
        
        # 3. Test Set
        full_decomp = self._decompose_and_pad(full_series)
        test_decomp_cd = full_decomp[border['start'][2]:border['end'][2]]
        #########################################################
        # Attention: Data Leakage
        # train_decomp = full_decomp[border['start'][0]:border['end'][0]]
        # val_decomp_cd = full_decomp[border['start'][1]:border['end'][1]]
        return {'train': train_decomp, 'val': val_decomp_cd, 'test': test_decomp_cd}

    def run(self):
        print(f"\nProcessing: {self.file_path}")
        df = pd.read_csv(self.file_path)
        
        cols_to_drop = [c for c in df.columns if c in ['date', 'Date', 'Time', 'time']]
        if cols_to_drop: df = df.drop(columns=cols_to_drop)
        # 确保全是数值
        df = df.select_dtypes(include=[np.number])
        data = df.values

        # 获取切分点
        total_len = len(df)
        border = self._get_borders(total_len)
        
        # ================== 修改部分开始 ==================
        # --- 数据标准化 (Fit on Train, Transform All) ---
        if self.scale:
            print("  > Executing Standardization (Fit on Train set)...")
            # 获取训练集的结束索引
            s0 = border['start'][0] # 通常是0，但为了严谨使用 border
            e0 = border['end'][0]
            s1 = border['start'][1]
            e1 = border['end'][1]
            s2 = border['start'][2]
            e2 = border['end'][2]
            # 1. 仅在训练集上 fit
            train_data_for_fit = data[s0:e0, :]
            self.scaler.fit(train_data_for_fit)
            print(f"  > Scaler fitted on range [{s0}:{e0}]")
            
            # 2. 对整个数据集 transform，后续 _process_column 切分出来的 train/val/test 都是归一化过的
            data = self.scaler.transform(data)
            print("  > Whole dataset transformed.")
        else:
            print("  > Skipping scaling (using raw data).")
        # ================== 修改部分结束 ==================
        print(f"  > Borders: Train[0:{border['end'][0]}], \n"
              f"  > Val[{border['start'][1]}:{border['end'][1]}], \n"
              f"  > Test[{border['start'][2]}:{border['end'][2]}]")
        
        # 此时 data_values 已经是处理过的数据，传入 _process_column 直接分解即可
        print(f"data_values shape: {data.shape}")
        
        results = Parallel(n_jobs=-1)(
            delayed(self._process_column)(data[:, i], border) 
            for i in tqdm(range(data.shape[1]), desc="Decomposing Cols")
        )
        
        # 堆叠
        train_list = [r['train'] for r in results]
        val_list   = [r['val'] for r in results]
        test_list  = [r['test'] for r in results]
        
        # 结果维度: [Time, Channel, K_IMFS]
        train_tensor = np.stack(train_list, axis=1)
        val_tensor =   np.stack(val_list, axis=1)
        test_tensor  = np.stack(test_list, axis=1)

        print(f"train_tensor: {train_tensor.shape}")
        print(f"val_tensor: {val_tensor.shape}")
        print(f"test_tensor: {test_tensor.shape}")

        base_name = self.file_path.replace('.csv', '')
        scale_suffix = "_scaled" if self.scale else "" 
        suffix = f"_sl{self.seq_len}{scale_suffix}_cd" 

        train_save_path = f"{base_name}_train{suffix}.npy"
        val_save_path = f"{base_name}_val{suffix}.npy"
        test_save_path = f"{base_name}_test{suffix}.npy"

        # Assert: Data and decomposed data are equal
        train_sum = train_tensor.sum(axis=-1) # [T, C]
        val_sum = val_tensor.sum(axis=-1) # [T, C]
        test_sum = test_tensor.sum(axis=-1) # [T, C]
        
        if np.allclose(data[s0:e0], train_sum) and np.allclose(data[s1:e1], val_sum) and np.allclose(data[s2:e2], test_sum):
            print("  > Data and decomposed data are equal.")
        else:
            print("Warning: Data and decomposed data are not equal, please check the decomposition.")
            exit(0)
        
        np.save(f"{base_name}_scaled.npy", data)
        np.save(train_save_path, train_tensor)
        np.save(val_save_path, val_tensor)
        np.save(test_save_path,  test_tensor)

        print(
            f"  > Saved Train {train_tensor.shape} to {train_save_path},\n" 
            f"  > Saved Val {val_tensor.shape} to {val_save_path},\n" 
            f"  > Saved Test {test_tensor.shape} to {test_save_path}\n"
            f"  > Saved Data {data.shape} to {base_name}_scaled.npy\n"
        )

class ShortTermDecomposition:
    def __init__(self, data_type, root_path, data_file, max_imfs=12, seq_len=96, scale=True):
        self.data_type = data_type
        self.root_path = root_path
        self.data_file = data_file
        self.file_path = os.path.join(root_path, data_file)
        self.max_imfs = max_imfs
        self.seq_len = seq_len
        self.scale = scale # 是否在分解前归一化，默认是归一化
        
        self.emd = EMD()
        self.emd.MAX_ITERATION = 200
        
        # 用于归一化的 Scaler
        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None # 不归一化
        print(f"   Decomposition: {self.file_path}\n"
              f"   max_imfs: {self.max_imfs}\n"
              f"   seq_len: {self.seq_len}\n"
              f"   scale: {self.scale}")

    def _decompose_and_pad(self, series_values):
        try:
            # EMD 分解
            # 注意：如果数据是恒定值，EMD会报错，需要Try-Catch
            imfs = self.emd.emd(series_values).T 
            T, n_imfs = imfs.shape
            
            result = np.zeros((T, self.max_imfs))
            if n_imfs >= self.max_imfs:
                result[:, :self.max_imfs-1] = imfs[:, :self.max_imfs-1]
                # 残差求和放入最后一个分量
                result[:, self.max_imfs-1] = np.sum(imfs[:, self.max_imfs-1:], axis=1)
            else:
                result[:, :n_imfs] = imfs
            return result
        except Exception as e:
            # print(f"Warning during decomposition: {e}")
            # 出错返回全0，避免程序中断
            return np.zeros((len(series_values), self.max_imfs))

    def _process_column(self, full_series, border):
        # 1. Train Set
        train_raw = full_series[border['start'][0]:border['end'][0]]
        train_decomp = self._decompose_and_pad(train_raw)
        
        # 2. Validation Set (Train + Val 的历史信息)
        val_raw = full_series[0:border['end'][1]] # 总是从 0 开始以保持索引对齐
        val_decomp_full = self._decompose_and_pad(val_raw)
        val_decomp_cd = val_decomp_full[border['start'][1]:border['end'][1]]
        
        # 3. Test Set (全部历史信息)
        full_decomp = self._decompose_and_pad(full_series)
        test_decomp_cd = full_decomp[border['start'][2]:border['end'][2]]
        # Attention: Data Leakage
        # train_decomp = full_decomp[border['start'][0]:border['end'][0]]
        # val_decomp_cd = full_decomp[border['start'][1]:border['end'][1]]
        return {'train': train_decomp, 'val': val_decomp_cd, 'test': test_decomp_cd}

    def run(self):
        print(f"\nProcessing: {self.file_path}")
        data = np.load(self.file_path, allow_pickle=True)
        data = data['data'][:, :, 0] # Use Flow
        # 1. 划分数据集索引
        train_ratio = 0.6
        valid_ratio = 0.2
        len_data = len(data)
        len_train = int(train_ratio * len(data))
        val_end = int((train_ratio + valid_ratio) * len_data)

        train_data = data[:len_train]
        border = {
            'start': [0,         len_train,  val_end],
            'end':   [len_train, val_end,    len_data]
        }

        # --- 数据标准化 (Fit on Train) ---
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
            np.save(f"{self.file_path.replace('.npz', '')}_scaled.npy", data)
            print("  > Data Scaled (Fit on Train set), warning: data will be scaled before decomposition.")
        
        print(f"  > Borders: Train[0:{border['end'][0]}], \n"
              f"  > Val[{border['start'][1]}:{border['end'][1]}], \n"
              f"  > Test[{border['start'][2]}:{border['end'][2]}]")

        # 并行处理
        results = Parallel(n_jobs=-1)(
            delayed(self._process_column)(data[:, i], border) 
            for i in tqdm(range(data.shape[1]), desc="Decomposing Cols")
        )
        
        # 堆叠
        train_list = [r['train'] for r in results]
        val_list   = [r['val'] for r in results]
        test_list  = [r['test'] for r in results]
        
        # 结果维度: [Time, Channel, K_IMFS]
        train_tensor = np.stack(train_list, axis=1)
        val_tensor =   np.stack(val_list, axis=1)
        test_tensor  = np.stack(test_list, axis=1)
        
        base_name = self.file_path.replace('.npz', '')
        suffix = f"_sl{self.seq_len}_scaled_cd"
        np.save(f"{base_name}_train{suffix}.npy", train_tensor)
        np.save(f"{base_name}_val{suffix}.npy", val_tensor)
        np.save(f"{base_name}_test{suffix}.npy",  test_tensor)
        
        print(f"  > Saved Train {train_tensor.shape} to {base_name}_train{suffix}.npy,\n" 
              f"  > Saved Val {val_tensor.shape} to {base_name}_val{suffix}.npy,\n" 
              f"  > Saved Test {test_tensor.shape} to {base_name}_test{suffix}.npy\n")
        
        # 验证：分解的3个分量之和等于原始信号：
        train_sum = train_tensor.sum(axis=-1) # [T, C]
        val_sum = val_tensor.sum(axis=-1) # [T, C]
        test_sum = test_tensor.sum(axis=-1) # [T, C]
        np.concatenate([train_sum, val_sum, test_sum], axis=0) # [L, C]
        # -> 验证data和分解的3个分量之和是否相等
        if np.allclose(data, np.concatenate([train_sum, val_sum, test_sum], axis=0)):
            print("  > Data and decomposed data are equal.")
            np.save(f"{base_name}_scaled.npy", data)
        else:
            print("  > Data and decomposed data are not equal, please check the decomposition.")
            exit(0)
 
       
def decompose_long_term_data(data_root, K_IMFS):
    DATA_LIST = [
        (f'{data_root}/ETT-small', 'ETTh', 'ETTh1.csv'),
        (f'{data_root}/ETT-small', 'ETTh', 'ETTh2.csv'),
        (f'{data_root}/ETT-small', 'ETTm', 'ETTm1.csv'),
        (f'{data_root}/ETT-small', 'ETTm', 'ETTm2.csv'),
        (f'{data_root}/electricity', 'custom', 'electricity.csv'),
        (f'{data_root}/exchange_rate','custom', 'exchange_rate.csv'),
        (f'{data_root}/weather', 'custom', 'weather.csv'),
        (f'{data_root}/traffic', 'custom', 'traffic.csv'),
        ]
    seq_lens = [96, 192, 336, 720]
    # shape [time_length, num_variables, num_imfs]
    for data_path, data_type, data_file in DATA_LIST:
        for seq_len in seq_lens:
            decomposer = LongTermDecomposition(
                data_type=data_type,
                root_path=data_path,
                data_file=data_file,
                max_imfs=K_IMFS,
                seq_len=seq_len)
            decomposer.run()
    
    ILL_DATA_LIST = [
        (f'{data_root}/illness', 'custom', 'national_illness.csv'),
    ]
    seq_lens = [24, 36, 48, 60] 
    for data_path, data_type, data_file in ILL_DATA_LIST:
        for seq_len in seq_lens:
            decomposer = LongTermDecomposition(
                data_type=data_type,
                root_path=data_path,
                data_file=data_file,
                max_imfs=K_IMFS,
                seq_len=seq_len)
            decomposer.run()
 
 
def decompose_short_term_data(data_root, K_IMFS):
    for data_file in ['PEMS03.npz', 'PEMS04.npz', 'PEMS07.npz', 'PEMS08.npz']:
        decomposer = ShortTermDecomposition(
            data_type='PEMS', 
            root_path=os.path.join(data_root, 'PEMS'),
            data_file=data_file,
            max_imfs=K_IMFS,
            seq_len=96)
        decomposer.run()
    
    
if __name__ == "__main__":
    K_IMFS = 15
    data_root = 'dataset'
    # decompose_long_term_data(data_root, K_IMFS)
    decompose_short_term_data(data_root, K_IMFS)
