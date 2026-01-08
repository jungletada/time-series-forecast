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

class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args, logger):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.logger = logger
        self.logger.info(f'Initializing Exp_Short_Term_Forecast.')
        
    def _build_model(self):
        if self.args.data_type == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2 * pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

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

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % self.args.print_freq == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            # --- Added Train Time per Epoch ---
            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # --- Added Peak GPU Memory Logic ---
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.logger.info("Epoch: {} Peak GPU memory: {:.2f} MB".format(epoch + 1, max_memory))
                torch.cuda.reset_peak_memory_stats()
            # -----------------------------------
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
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

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            self.logger.info('loading model')
            ckpt_path = os.path.join(self.args.checkpoints, setting['save_dir']+'_'+self.args.seasonal_patterns)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth')))

        result_path = os.path.join(self.args.results, setting['save_dir'])
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # --- Added Initialization ---
        inference_time = 0
        # ----------------------------

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            # --- Added Timing Start ---
            start_time = time.time()
            # --------------------------
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]], 
                    None)

                # if id_list[i] % 1000 == 0:
                #     self.logger.info(f"Processing batch {id_list[i]} of {B}")

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()
            # --- Added Timing End ---
            inference_time += time.time() - start_time
            # ------------------------
            preds = outputs
            x = x.detach().cpu().numpy()
            # trues = y
            # for i in range(0, preds.shape[0], preds.shape[0] // 10):
            #     gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
            #     pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
            #     visual(gt, pd, os.path.join(result_path, str(i) + '.pdf'))

        self.logger.info(f'test shape: {preds.shape}')

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(os.path.join(result_path, f'{self.args.seasonal_patterns}_forecast.csv'))

        # self.logger.info(f'{self.args.model}')
        m4_files = [f'{sp}_forecast.csv' for sp in ['Weekly', 'Monthly', 'Yearly', 'Daily', 'Hourly', 'Quarterly']]
        if all(f in os.listdir(result_path) for f in m4_files):
            m4_summary = M4Summary(result_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            self.logger.info(f'smape: {smape_results}')
            self.logger.info(f'mape: {mape}')
            self.logger.info(f'mase: {mase}')
            self.logger.info(f'owa: {owa_results}')
        else:
            self.logger.info('After all 6 tasks are finished, you can calculate the averaged index')
        return
