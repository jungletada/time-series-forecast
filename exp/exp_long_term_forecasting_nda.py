import os
import time
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.metrics import metric
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.dtw_metric import dtw, accelerated_dtw
# from utils.augmentation import run_augmentation, run_augmentation_single
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecas_NDA(Exp_Basic):
    def __init__(self, args, logger):
        super(Exp_Long_Term_Forecas_NDA, self).__init__(args)
        self.logger = logger
        self.logger.info(f'>>>>>>>>>>>>>> Initializing Exp_Long_Term_Forecas_NDA using Neural Decomposition Adapter. \n')
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # AdamW is used just like Adam, but it decouples weight decay from the gradient update.
        # Example usage:
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),   
            eps=1e-8                     
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 
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

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for iter_step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # batch_x/y shape: [B, T, C, K+1]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # ============================================================
                # 1. 切分数据 (Slicing)
                # ============================================================
                # 输入 Input: 取后 K 个分解分量 (Indices 1 to End)
                # Shape: [B, Seq_Len, C, K]
                model_input = batch_x[:, :, :, 1:] 

                # 目标 Target: 取第 0 个分量 (原始信号 Original Signal)
                # Shape: [B, Label_Len + Pred_Len, C]
                batch_y_target = batch_y[:, :, :, 0] 
                # ============================================================

                # 构造 decoder input (基于原始信号)
                dec_inp = torch.zeros_like(batch_y_target[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_target[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                moe_loss = 0.0
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        # Model 直接输出原始信号预测 [B, Pred_Len, C]
                        outputs = self.model(model_input, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple):
                            outputs, moe_loss = outputs
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_loss = batch_y_target[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        loss = criterion(outputs, batch_y_loss) + moe_loss * self.args.moe_weight
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(model_input, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple):
                        outputs, moe_loss = outputs
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_loss = batch_y_target[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, batch_y_loss) + moe_loss * self.args.moe_weight
                    train_loss.append(loss.item())

                if (iter_step + 1) % self.args.print_freq == 0:
                    self.logger.info("\t iters: {0}, epoch: {1} | loss: {2:.7f}".format(iter_step + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iter_step)
                    self.logger.info('\t speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.logger.info("Epoch: {} Peak GPU memory: {:.2f} MB".format(epoch + 1, max_memory))
                torch.cuda.reset_peak_memory_stats()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, ckpt_path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping...")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 1. 切分: 输入分解分量，预测原始信号
                model_input = batch_x[:, :, :, 1:]
                batch_y_target = batch_y[:, :, :, 0]

                dec_inp = torch.zeros_like(batch_y_target[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_target[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(model_input, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(model_input, batch_x_mark, dec_inp, batch_y_mark)
                
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_loss = batch_y_target[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y_loss.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.checkpoints, setting['save_dir'], 'checkpoint.pth')))

        preds = []
        trues = []
        result_path = os.path.join(self.args.results, setting['save_dir'])
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.model.eval()
        inference_time = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 1. 切分
                model_input = batch_x[:, :, :, 1:] # Input: Decomp
                batch_y_target = batch_y[:, :, :, 0] # Target: Original

                dec_inp = torch.zeros_like(batch_y_target[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_target[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                start_time = time.time()
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(model_input, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(model_input, batch_x_mark, dec_inp, batch_y_mark)
                
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                    
                inference_time += time.time() - start_time

                f_dim = -1 if self.args.features == 'MS' else 0
                
                # Slicing
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_target = batch_y_target[:, -self.args.pred_len:, :]
                
                # CPU Transfer
                outputs = outputs.detach().cpu().numpy()
                batch_y_target = batch_y_target.detach().cpu().numpy()

                # Inverse Transform
                # 此时 Target 和 Output 都是原始信号维度 [B, L, C]，直接 transform 即可
                if test_data.scale and self.args.inverse:
                    shape = batch_y_target.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, shape[-1])).reshape(shape)
                    batch_y_target = test_data.inverse_transform(batch_y_target.reshape(-1, shape[-1])).reshape(shape)

                # Metric 选取
                if self.args.features == 'MS':
                    pred = outputs[:, :, -1:]
                    true = batch_y_target[:, :, -1:]
                else:
                    pred = outputs
                    true = batch_y_target

                preds.append(pred)
                trues.append(true)
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        self.logger.info(f'test shape: {preds.shape}, {trues.shape}')
        avg_latency = (inference_time / (i + 1)) * 1000
        self.logger.info("Average Inference Latency: {:.2f} ms/batch".format(avg_latency))

        # Metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.logger.info('mse:{:.5f}, mae:{:.5f}, rmse:{:.5f}, mape:{:.5f}, mspe:{:.5f}'.format(mse, mae, rmse, mape, mspe))

        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_path, 'pred.npy'), preds)
        np.save(os.path.join(result_path, 'true.npy'), trues)

        return
    