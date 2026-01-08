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

class Dep_Short_Term_Forecasting(Exp_Basic):
    def __init__(self, args, logger):
        # 定义分量名称，用于日志打印
        self.comp_names = ['High', 'Mid', 'Low']
        self.num_components = len(self.comp_names)

        super(Dep_Short_Term_Forecasting, self).__init__(args)
        self.logger = logger
        self.logger.info(f'Initializing Exp_Dep_Short_Term_Forecasting.')
        self.logger.info(f'Number of components: {self.num_components}')

    def _build_model(self):
        if 'M4' in self.args.data_type:
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
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        ckpt_path = os.path.join(self.args.checkpoints, setting['save_dir']+'_'+self.args.seasonal_patterns)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            for m in self.model: m.train()
            epoch_time = time.time()
            
            # Batch Data: [Batch, 3, Seq_Len, 1]
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                total_loss = 0
                
                # --- 核心训练循环：遍历 3 个分量 ---
                for k in range(self.num_components):
                    # 1. 提取第 k 个分量的数据
                    # Shape: [Batch, Seq_Len, 1]
                    inp_k = batch_x[:, k, :, :]
                    target_k = batch_y[:, k, :, :]
                    batch_y_mark_k = batch_y_mark[:, k, :, :]
                    # 2. 构造 Decoder Input (M4 常用 Label Len 拼接)
                    dec_inp = torch.zeros_like(target_k[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([target_k[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    # 3. 第 k 个模型前向传播
                    # 注意：Time-Series-Library 模型通常接收 (x, x_mark, dec_inp, y_mark)
                    outputs = self.model[k](inp_k, None, dec_inp, None)

                    # 4. 截取预测部分
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    target_k_pred = target_k[:, -self.args.pred_len:, f_dim:].to(self.device)
                    mark_k = batch_y_mark_k[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # 5. 计算 Component Loss
                    # 注意：criterion 通常需要 input (history) 用于计算缩放因子 (如 MASE/SMAPE)
                    # insample: t.Tensor, freq: int, forecast: t.Tensor, target: t.Tensor, mask: t.Tensor
                    loss_k = criterion(inp_k, self.args.frequency_map, outputs, target_k_pred, mark_k)
                    
                    # 6. 累加 Loss
                    total_loss += loss_k

                train_loss.append(total_loss.item())

                if (i + 1) % self.args.print_freq == 0:
                    self.logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, total_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                total_loss.backward()
                model_optim.step()

            # --- Added Train Time per Epoch ---
            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # --- Added Peak GPU Memory Logic ---
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.logger.info("Epoch: {} Peak GPU memory: {:.2f} MB".format(epoch + 1, max_memory))
                torch.cuda.reset_peak_memory_stats()
            
            train_loss = np.average(train_loss)
            # --- Validation ---
            # 验证时，我们评估的是"重构后"的信号精度
            # vali_loss = self.vali(train_loader, vali_loader, criterion)
            self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            
            # early_stopping(vali_loss, self.model, ckpt_path)
            # if early_stopping.early_stop:
            #     self.logger.info("Early stopping...")
            #     break
            torch.save(self.model.state_dict(), best_model_path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    # def vali(self, train_loader, vali_loader, criterion):
    #     # 1. 获取输入 x (Last Insample Window of Train Data)
    #     # x: [N, 3, Seq_Len, 1]
    #     x, _ = train_loader.dataset.last_insample_window()
    #     x = torch.tensor(x, dtype=torch.float32).to(self.device)

    #     # 2. 获取正确的 GT (Test Ground Truth)
    #     # 错误做法: y = vali_loader.dataset.timeseries (这是变长的历史训练数据!)
    #     # 正确做法: 手动加载生成的 test_cd.npy，它包含未来真值
        
    #     # 构造 test_cd.npy 的路径
    #     freq = self.args.seasonal_patterns
    #     test_npy_path = os.path.join(self.args.root_path, f"M4_{freq}_test_cd.npy")
        
    #     if not os.path.exists(test_npy_path):
    #         raise FileNotFoundError(f"Test GT file not found: {test_npy_path}")
            
    #     # 加载 Test GT (List of [Pred_Len, K])
    #     test_decomp_list = np.load(test_npy_path, allow_pickle=True)
        
    #     # 3. 预处理 GT: 求和还原为原始信号 & 转换为 Tensor
    #     # M4 的 Test set 长度是固定的 (例如 Hourly=48)，所以可以 Stack
    #     y_raw_list = []
    #     for ts in test_decomp_list:
    #         # ts shape: [Pred_Len, K] -> Sum -> [Pred_Len, 1]
    #         # 注意: 如果你的 decomposition 代码返回的是 [K, Pred_Len]，这里要做对应调整
    #         # 根据之前的代码，test_cd.npy 保存的是 [Pred_Len, K]，所以 axis=1 求和
    #         # 如果之前的代码返回的是 [K, Pred_Len]，则 axis=0
    #         # 假设之前代码是 [Pred_Len, K]:
    #         if ts.shape[0] == self.args.pred_len:
    #             raw = ts.sum(axis=1).reshape(-1, 1)
    #         else:
    #             # 兼容性处理，防止维度转置
    #             raw = ts.sum(axis=0).reshape(-1, 1)
    #         y_raw_list.append(raw)
            
    #     # Stack -> [N, Pred_Len, 1]
    #     y_true_tensor = torch.from_numpy(np.stack(y_raw_list)).float().to(self.device)

    #     # 校验长度
    #     assert len(x) == len(y_true_tensor), f"Input size {len(x)} != GT size {len(y_true_tensor)}"

    #     self.model.eval()
    #     with torch.no_grad():
    #         B, Num_Comps, Seq_Len, C = x.shape
    #         final_preds = []
            
    #         # 分批次推理
    #         id_list = np.arange(0, B, 500)
    #         id_list = np.append(id_list, B)
            
    #         for i in range(len(id_list) - 1):
    #             batch_x_slice = x[id_list[i]:id_list[i + 1]] 
    #             batch_pred_sum = 0
                
    #             for k in range(self.num_components):
    #                 inp_k = batch_x_slice[:, k, :, :]
                    
    #                 # 构造 Decoder Input (用历史输入的最后部分)
    #                 dec_inp = torch.zeros((len(inp_k), self.args.pred_len, C)).float().to(self.device)
    #                 dec_inp = torch.cat([inp_k[:, -self.args.label_len:, :], dec_inp], dim=1).float()
                    
    #                 output_k = self.model[k](inp_k, None, dec_inp, None)
    #                 batch_pred_sum += output_k
                
    #             final_preds.append(batch_pred_sum) # 保持在 GPU 上以节省传输时间
                
    #         # 拼接 -> [N, Pred_Len, C]
    #         outputs = torch.cat(final_preds, dim=0)
            
    #         # 截取预测部分
    #         f_dim = -1 if self.args.features == 'MS' else 0
    #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
            
    #         # 4. 计算 Metric
    #         # pred: [N, Pred_Len, 1]
    #         pred = outputs 
    #         # true: [N, Pred_Len, 1] (现在它是 Tensor 了!)
    #         true = y_true_tensor 
            
    #         # Input Sum (用于 M4 Loss 的缩放因子)
    #         input_x_raw = x.sum(dim=1) # [N, Seq, 1]
            
    #         # Dummy Mark
    #         batch_y_mark = torch.ones_like(true)

    #         # 现在的 criterion 输入全是 Tensor，不会报错了
    #         loss = criterion(
    #             input_x_raw[:, :, 0], 
    #             self.args.frequency_map, 
    #             pred[:, :, 0], 
    #             true[:, :, 0], 
    #             batch_y_mark[:, :, 0]
    #         )

    #     self.model.train()
    #     return loss
    
    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        
        # 1. 获取输入 [N, 3, Seq, 1]
        x, _ = train_loader.dataset.last_insample_window()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        if test:
            self.logger.info('loading model')
            ckpt_path = os.path.join(self.args.checkpoints, setting['save_dir']+'_'+self.args.seasonal_patterns)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth')))

        result_path = os.path.join(self.args.results, setting['save_dir'])
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.model.eval()
        with torch.no_grad():
            B, Num_Comps, Seq_Len, C = x.shape
            
            final_outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            
            id_list = np.arange(0, B, 1) # M4 测试集较小，通常可以逐个或小批次
            id_list = np.append(id_list, B)
            
            start_time = time.time()
            
            for i in range(len(id_list) - 1):
                batch_x_slice = x[id_list[i]:id_list[i + 1]]
                batch_sum_k = 0
                
                # --- 分量推理与重构 ---
                for k in range(self.num_components):
                    inp_k = batch_x_slice[:, k, :, :]
                    
                    dec_inp = torch.zeros((len(inp_k), self.args.pred_len, C)).float().to(self.device)
                    dec_inp = torch.cat([inp_k[:, -self.args.label_len:, :], dec_inp], dim=1).float()
                    
                    out_k = self.model[k](inp_k, None, dec_inp, None)
                    batch_sum_k += out_k
                
                final_outputs[id_list[i]:id_list[i + 1], :, :] = batch_sum_k

            # 后处理
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = final_outputs[:, -self.args.pred_len:, f_dim:]
            preds = outputs.detach().cpu().numpy() # [N, Pred_Len, 1]
            
            print(f"Inference Time: {time.time() - start_time:.4f}s")

        self.logger.info(f'test shape: {preds.shape}')

        # M4 输出格式化 (保持原逻辑)
        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(os.path.join(result_path, f'{self.args.seasonal_patterns}_forecast.csv'))

        # M4 Summary Evaluation (保持原逻辑)
        m4_files = [f'{sp}_forecast.csv' for sp in ['Weekly', 'Monthly', 'Yearly', 'Daily', 'Hourly', 'Quarterly']]
        if all(f in os.listdir(result_path) for f in m4_files):
            m4_summary = M4Summary(result_path, self.args.root_path)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            self.logger.info(f'smape: {smape_results}')
            self.logger.info(f'mape: {mape}')
            self.logger.info(f'mase: {mase}')
            self.logger.info(f'owa: {owa_results}')
        else:
            self.logger.info('After all 6 tasks are finished, you can calculate the averaged index')
        return