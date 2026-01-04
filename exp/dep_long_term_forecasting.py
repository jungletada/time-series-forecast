import os
import time
import warnings
import numpy as np
import json
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dtw_metric import dtw, accelerated_dtw

warnings.filterwarnings('ignore')

class Exp_Dep_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args, logger):
        # 定义分量名称，用于日志打印
        self.comp_names = ['High', 'Mid', 'Low']
        self.num_components = len(self.comp_names)

        # 调用父类初始化
        super(Exp_Dep_Long_Term_Forecast, self).__init__(args)
        self.logger = logger
        self.logger.info(f'Initializing Exp_Dep_Long_Term_Forecast (Training K={self.num_components} Independent Models).')
        self.logger.info(f'Number of components: {self.num_components}')

    def _build_model(self):
        # 覆盖父类方法：我们需要构建 3 个独立的模型
        # 注意：这里假设 3 个模型使用相同的架构 (args.model)
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
        # 为每个模型创建一个独立的优化器
        optimizers = []
        for model in self.model: # self.model 现在是一个 ModuleList
            model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
            optimizers.append(model_optim)
        return optimizers

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, model_idx):
        """
        辅助函数：处理单个模型的 Forward
        """
        # 数据已经是 [B, 3, T, C]，我们取对应的 model_idx 分量 -> [B, T, C]
        b_x = batch_x[:, model_idx, :, :].float().to(self.device)
        b_y = batch_y[:, model_idx, :, :].float().to(self.device)
        
        b_x_mark = batch_x_mark.float().to(self.device)
        b_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(b_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([b_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = self.model[model_idx](b_x, b_x_mark, dec_inp, b_y_mark)
        else:
            outputs = self.model[model_idx](b_x, b_x_mark, dec_inp, b_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        b_y = b_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, b_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # 切换所有模型到 Eval 模式
        for m in self.model: m.eval()
            
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                loss_sum = 0
                # 对 3 个分量分别预测并计算 Loss
                for comp_idx in range(self.num_components):
                    pred, true = self._process_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark, comp_idx
                    )
                    loss = criterion(pred.detach(), true.detach())
                    loss_sum += loss.item()
                
                # 记录平均 Loss (或者总 Loss)
                total_loss.append(loss_sum / self.num_components)
                
        total_loss = np.average(total_loss)
        # 切换回 Train 模式
        for m in self.model: m.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        ckpt_path = os.path.join(self.args.checkpoints, setting['save_dir'])
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optimizers = self._select_optimizer() # List of K optimizers
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            # Ensure all models are in train mode
            for m in self.model: m.train()
            
            epoch_time = time.time()
            for iter_step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                # Zero Grad for all optimizers
                for opt in model_optimizers: opt.zero_grad()
                
                batch_total_loss = 0
                
                # --- Independent Forward & Backward ---
                for comp_idx in range(self.num_components):
                    outputs, true_y = self._process_one_batch(
                        batch_x=batch_x, 
                        batch_y=batch_y, 
                        batch_x_mark=batch_x_mark, 
                        batch_y_mark=batch_y_mark, 
                        model_idx=comp_idx
                    )
                    
                    loss = criterion(outputs, true_y)
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    batch_total_loss += loss.item()
                
                # Step all optimizers
                if self.args.use_amp:
                    for opt in model_optimizers:
                        scaler.step(opt)
                    scaler.update()
                else:
                    for opt in model_optimizers:
                        opt.step()

                train_loss.append(batch_total_loss / self.num_components)

                if (iter_step + 1) % self.args.print_freq == 0:
                    self.logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        iter_step + 1, epoch + 1, batch_total_loss / self.num_components))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iter_step)
                    self.logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # --- Added Train Time per Epoch ---
            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # --- Added Peak GPU Memory Logic ---
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.logger.info("Epoch: {} Peak GPU memory: {:.2f} MB".format(epoch + 1, max_memory))
                torch.cuda.reset_peak_memory_stats()
            # -----------------------------------
            # 保存逻辑：我们保存整个 ModuleList
            early_stopping(vali_loss, self.model, ckpt_path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping...")
                break

            for opt in model_optimizers:
                adjust_learning_rate(opt, epoch + 1, self.args)

        # 加载最优模型
        best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting['save_dir'], 'checkpoint.pth')))

        preds = []
        trues = []
        
        result_path = os.path.join(self.args.results, setting['save_dir'])
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # --- Added Initialization ---
        inference_time = 0
        # ----------------------------
            
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                
                # 存储当前 Batch 的 3 个分量预测结果
                batch_preds_list = []
                batch_trues_list = []
                # --- Added Timing Start ---
                start_time = time.time()
                # --------------------------
                for comp_idx in range(self.num_components):
                    # 获取单分量预测
                    outputs, true_y = self._process_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark, comp_idx
                    )
                    # outputs: [B, Pred_Len, C]
                    batch_preds_list.append(outputs.detach().cpu().numpy())
                    batch_trues_list.append(true_y.detach().cpu().numpy())
                
                # --- 核心：聚合 (Aggregation) ---
                # 将 3 个分量相加 -> 还原为归一化的原始信号
                # [B, Pred_Len, C]
                pred_sum = np.sum(batch_preds_list, axis=0) 
                true_sum = np.sum(batch_trues_list, axis=0)
                # --- Added Timing End ---
                inference_time += time.time() - start_time
                # ------------------------
                # --- 反归一化 (Inverse Transform) ---
                if test_data.scale and self.args.inverse:
                    shape = pred_sum.shape
                    # 如果输出列数不匹配 (features='M' vs 'MS')，做 tile 处理
                    if pred_sum.shape[-1] != true_sum.shape[-1]: 
                        # 这种情况通常不需要处理，因为我们是 sum 后的
                        pass 
                    
                    # 这里的 reshape 是为了适配 scaler 的输入
                    pred_sum = test_data.inverse_transform(pred_sum.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true_sum = test_data.inverse_transform(true_sum.reshape(shape[0] * shape[1], -1)).reshape(shape)

                preds.append(pred_sum)
                trues.append(true_sum)

                # 可视化 (Visual)
                if i % 20 == 0:
                    # 为了可视化 Input，我们也需要对 Input 的 3 分量求和
                    # Input: [B, 3, Seq_Len, C] -> Sum dim1 -> [B, Seq_Len, C]
                    input_x = batch_x.sum(dim=1).detach().cpu().numpy()
                    
                    if test_data.scale and self.args.inverse:
                        shape = input_x.shape
                        input_x = test_data.inverse_transform(input_x.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    
                    gt = np.concatenate((input_x[0, :, -1], true_sum[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], pred_sum[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(result_path, str(i) + '.pdf'))

        # Concatenate all batches
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        self.logger.info(f'Final Test Shape: {preds.shape}, {trues.shape}')

        # --- Added Latency Printing ---
        avg_latency = (inference_time / (i + 1)) * 1000
        self.logger.info("Average Inference Latency: {:.2f} ms/batch".format(avg_latency))
        # ------------------------------

        # Metrics Calculation
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.logger.info('mse:{:.5f}, mae:{:.5f}, rmse:{:.5f}, mape:{:.5f}, mspe:{:.5f}'.format(mse, mae, rmse, mape, mspe))
        
        # Save results
        f = open(os.path.join(result_path, 'result_dep_long_term_forecast.txt'), 'a')
        f.write(json.dumps(setting) + "  \n")
        f.write('mse:{:.5f}, mae:{:.5f}, rmse:{:.5f}, mape:{:.5f}, mspe:{:.5f}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_path, 'pred.npy'), preds)
        np.save(os.path.join(result_path, 'true.npy'), trues)
        
        return