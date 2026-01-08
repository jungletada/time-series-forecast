import os
import numpy as np
import pandas as pd
from PyEMD import EMD
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from PyEMD import EMD
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class M4Decomposition:
    def __init__(self, root_path, frequency, max_imfs=10):
        self.root_path = root_path
        self.frequency = frequency
        self.max_imfs = max_imfs
        self.emd = EMD()
        self.emd.MAX_ITERATION = 100
        
    def _decompose_single_series(self, series_values, return_tail_len=None):
        """
        对单条序列进行分解
        args:
            series_values: 输入的序列数值 (numpy array)
            return_tail_len: 如果不为None，则只返回最后 return_tail_len 长度的分解结果
                             (用于 Test 模式，只取分解后的未来部分)
        """
        # 1. 去除 NaN (M4 Train CSV有填充的空值)
        valid_values = series_values[~np.isnan(series_values)]
        current_len = len(valid_values)
        
        # 确定需要返回的长度
        if return_tail_len is not None:
            # 如果是 Test 模式，我们需要返回的长度是固定的 Horizon
            target_len = return_tail_len
        else:
            # 如果是 Train 模式，返回全部有效长度
            target_len = current_len

        # 2. 极短序列处理 (Fallback)
        # 如果序列太短无法分解，或者比要截取的长度还短
        if current_len < 4 or current_len < target_len: 
            res = np.zeros((target_len, self.max_imfs))
            # 这种情况下很难办，简单处理：
            # 如果是 Test 模式且序列极短，填入原始值的最后部分
            res[:, 0] = valid_values[-target_len:] if target_len <= current_len else 0
            return res

        try:
            # 3. EMD 分解
            imfs = self.emd.emd(valid_values).T 
            T_actual, n_imfs = imfs.shape
            
            # 4. 整理分量
            processed_imfs = np.zeros((T_actual, self.max_imfs))
            if n_imfs >= self.max_imfs:
                processed_imfs[:, :self.max_imfs-1] = imfs[:, :self.max_imfs-1]
                processed_imfs[:, self.max_imfs-1] = np.sum(imfs[:, self.max_imfs-1:], axis=1)
            else:
                processed_imfs[:, :n_imfs] = imfs            
            
            # 5. 截取逻辑 (关键修改)
            if return_tail_len is not None:
                # Test 模式：只返回最后 N 个点 (即 Test 部分的分解结果)
                # 这样我们就得到了基于完整历史上下文分解出来的"未来分量"
                return processed_imfs[-return_tail_len:, :]
            else:
                # Train 模式：返回全部分解结果
                return processed_imfs
                
        except Exception as e:
            # Fallback
            fallback = np.zeros((target_len, self.max_imfs))
            fallback[:, 0] = valid_values[-target_len:] if target_len <= current_len else 0
            return fallback

    def run(self):
        # 1. 定义文件路径
        train_path = os.path.join(self.root_path, f"{self.frequency}-train.csv")
        test_path = os.path.join(self.root_path, f"{self.frequency}-test.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Skipping {self.frequency}, files not found.")
            return

        print(f"\nProcessing {self.frequency}...")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        # 确保对齐
        assert np.all(df_train.iloc[:, 0].values == df_test.iloc[:, 0].values)
        
        train_vals = df_train.iloc[:, 1:].values
        test_vals = df_test.iloc[:, 1:].values
        
        # M4 的 Test 文件通常是固定宽度的 (即 Horizon 长度固定)
        # 例如 Hourly 都是 48，Yearly 都是 6
        # 我们假设每一行的 Test 长度都是一样的 (这在 M4 中是成立的)
        test_horizon = test_vals.shape[1]
        
        print(f"  > Series Count: {len(train_vals)}")
        print(f"  > Test Horizon: {test_horizon}")

        # ==========================================
        # Phase 1: 处理 Train (仅使用历史数据)
        # ==========================================
        print(f"  > Decomposing TRAIN set (History only)...")
        train_results = Parallel(n_jobs=-1)(
            delayed(self._decompose_single_series)(train_vals[i], return_tail_len=None) 
            for i in tqdm(range(len(train_vals)), desc="Train Decomp")
        )
        
        # 保存 Train
        train_obj = np.array(train_results, dtype=object)
        np.save(os.path.join(self.root_path, f"M4_{self.frequency}_train_cd.npy"), train_obj, allow_pickle=True)
        
        # ==========================================
        # Phase 2: 处理 Test (历史 + 未来 -> 分解 -> 截取未来)
        # ==========================================
        print(f"  > Decomposing TEST set (History + Future -> Slice Future)...")
        
        # 拼接数据：每一行 = Train有效部分 + Test部分
        # 注意：不能直接 np.concatenate，因为 train_vals 里有 NaN
        # 我们需要在并行函数里处理拼接，或者这里预处理一下
        # 为了简单，我们在 _decompose_single_series 内部逻辑是接收一个完整数组
        # 所以我们需要在这里把 train 和 test 拼起来传递进去
        
        full_seqs = []
        for tr, te in zip(train_vals, test_vals):
            # 去除 train 的 NaN (padding)
            valid_tr = tr[~np.isnan(tr)]
            # 拼接
            full_seqs.append(np.concatenate([valid_tr, te]))
        
        # 并行分解 (传入 return_tail_len = test_horizon)
        test_results = Parallel(n_jobs=-1)(
            delayed(self._decompose_single_series)(full_seqs[i], return_tail_len=test_horizon) 
            for i in tqdm(range(len(full_seqs)), desc="Test Decomp")
        )
        
        # 保存 Test (此时里面只包含 Test 时间段的分解分量，但是是基于全局分解得到的)
        test_obj = np.array(test_results, dtype=object)
        np.save(os.path.join(self.root_path, f"M4_{self.frequency}_test_cd.npy"), test_obj, allow_pickle=True)
        
        print(f"  > Done. Saved train_cd.npy and test_cd.npy")
        
class CompleteDecomposition:
    def __init__(self, data_type, root_path, data_file, max_imfs=10, seq_len=96, scale=False):
        self.data_type = data_type
        self.root_path = root_path
        self.data_file = data_file
        self.file_path = os.path.join(root_path, data_file)
        self.max_imfs = max_imfs
        self.seq_len = seq_len
        self.scale = scale # 是否在分解前归一化，默认不归一化
        
        self.emd = EMD()
        # 针对 ETTm 等长序列，适当放宽迭代限制或使用 CEEMDAN 可能更好，但这里保持原样
        self.emd.MAX_ITERATION = 100 
        
        # 用于归一化的 Scaler
        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None # 不归一化
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
        # 此时 val_decomp_full 的长度为 border['end'][1]
        # 我们切取出 [border['start'][1] : border['end'][1]]
        val_decomp_cd = val_decomp_full[border['start'][1]:border['end'][1]]
        
        # 3. Test Set (全部历史信息)
        full_decomp = self._decompose_and_pad(full_series)
        test_decomp_cd = full_decomp[border['start'][2]:border['end'][2]]
        
        return {'train': train_decomp, 'val': val_decomp_cd, 'test': test_decomp_cd}

    def run(self):
        print(f"\nProcessing: {self.file_path}")
        df = pd.read_csv(self.file_path)
        
        cols_to_drop = [c for c in df.columns if c in ['date', 'Date', 'Time', 'time']]
        if cols_to_drop: df = df.drop(columns=cols_to_drop)
        # 确保全是数值
        df = df.select_dtypes(include=[np.number])
        data_values = df.values

        # 获取切分点
        total_len = len(df)
        border = self._get_borders(total_len)
        
        # --- 数据标准化 (Fit on Train) ---
        if self.scale:
            train_end = border['end'][0]
            self.scaler.fit(data_values[:train_end])
            data_values = self.scaler.transform(data_values)
            print("  > Data Scaled (Fit on Train set), warning: data will be scaled after decomposition.")

        print(f"  > Borders: Train[0:{border['end'][0]}], \n"
              f"  > Val[{border['start'][1]}:{border['end'][1]}], \n"
              f"  > Test[{border['start'][2]}:{border['end'][2]}]")
        
        # 并行处理
        # 这里的 data_values 已经是 numpy array
        results = Parallel(n_jobs=-1)(
            delayed(self._process_column)(data_values[:, i], border) 
            for i in tqdm(range(data_values.shape[1]), desc="Decomposing Cols")
        )
        
        # 堆叠
        train_list = [r['train'] for r in results]
        val_list   = [r['val'] for r in results]
        test_list  = [r['test'] for r in results]
        
        # 结果维度: [Time, Channel, K_IMFS]
        train_tensor = np.stack(train_list, axis=1)
        val_tensor =   np.stack(val_list, axis=1)
        test_tensor  = np.stack(test_list, axis=1)
        
        base_name = self.file_path.replace('.csv', '')
        # 建议文件名带上 seq_len 防止混淆
        suffix = f"_sl{self.seq_len}_cd"
        np.save(f"{base_name}_train{suffix}.npy", train_tensor)
        np.save(f"{base_name}_val{suffix}.npy", val_tensor)
        np.save(f"{base_name}_test{suffix}.npy",  test_tensor)
        
        print(f"  > Saved Train {train_tensor.shape} to {base_name}_train{suffix}.npy,\n" 
              f"  > Saved Val {val_tensor.shape} to {base_name}_val{suffix}.npy,\n" 
              f"  > Saved Test {test_tensor.shape} to {base_name}_test{suffix}.npy\n")


def decompose_long_term_data(data_root, K_IMFS):
    DATA_LIST = [
        (f'{data_root}/ETT-small', 'ETTh', 'ETTh1.csv'),
        (f'{data_root}/ETT-small', 'ETTh', 'ETTh2.csv'),
        (f'{data_root}/ETT-small', 'ETTm', 'ETTm1.csv'),
        (f'{data_root}/ETT-small', 'ETTm', 'ETTm2.csv'),
        (f'{data_root}/electricity', 'custom', 'electricity.csv'),
        (f'{data_root}/weather', 'custom', 'weather.csv'),
        (f'{data_root}/exchange_rate','custom', 'exchange_rate.csv'),
        (f'{data_root}/traffic', 'custom', 'traffic.csv'),
        ]
    seq_lens = [96, 192, 336, 720]
    # shape [time_length, num_variables, num_imfs]
    for data_path, data_type, data_file in DATA_LIST:
        for seq_len in seq_lens:
            decomposer = CompleteDecomposition(
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
            decomposer = CompleteDecomposition(
                data_type=data_type,
                root_path=data_path,
                data_file=data_file,
                max_imfs=K_IMFS,
                seq_len=seq_len)
            decomposer.run()
 

def decompose_short_term_data(data_root, K_IMFS):
     # M4 数据集路径
    M4_ROOT = os.path.join(data_root, 'm4') 
    # M4 的 6 个子集
    SUBSETS = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
    
    # 注意：M4 分解不需要指定 seq_len 参数，
    # 因为我们是对整个历史序列做分解。
    # seq_len 是下游模型读取数据时采样的窗口大小。
    
    for freq in SUBSETS:
        decomposer = M4Decomposition(
            root_path=M4_ROOT,
            frequency=freq,
            max_imfs=K_IMFS,
        )
        decomposer.run()


if __name__ == "__main__":
    K_IMFS = 10
    data_root = 'dataset'
    # decompose_long_term_data(data_root, K_IMFS)
    decompose_short_term_data(data_root, K_IMFS)