import os
import time
import warnings
import numpy as np
import json
import torch
import torch.nn as nn
from torch import optim
import torch.multiprocessing as mp # 确保导入了多进程库
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.loss import FirstOrderDiffLoss, LogCoshLoss
from utils.dtw_metric import dtw, accelerated_dtw

warnings.filterwarnings('ignore')

class Exp_Dep_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args, logger):
        # 定义分量名称，用于日志打印
        self.logger = logger
        self.logger.info(f'Initializing Exp_Long_Term_Forecast (Training {args.num_imf} Independent Models).')
        self.logger.info(f'Number of components: {args.num_imf}')
        super(Exp_Dep_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        models = []
        # 严格依赖 model_args_list
        for i, model_args in enumerate(self.args.model_args_list):
            self.logger.info(f'Building model {i} with specific args: {model_args}')
            
            # 使用各自的配置实例化
            model = self.model_dict[model_args.model].Model(model_args).float()
            
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            models.append(model)
            
        return nn.ModuleList(models)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        print(f">>>>>>>>>>>>>>>>>>>>>> length of data_set: {len(data_set)}")
        return data_set, data_loader

    def _select_optimizer(self):
        optimizers = []
        for i, model_args in enumerate(self.args.model_args_list):
            model = self.model[i]
            lr = model_args.learning_rate
            model_optim = optim.Adam(model.parameters(), lr=lr)
            optimizers.append(model_optim)
            
            self.logger.info(f"Built Optimizer for Component-{i} with lr={lr}")
            
        return optimizers

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _train_single_component(self, model, model_args, component_idx, setting, train_loader, vali_loader, test_loader):
        """
        独立训练单个分量的函数
        """
        # 1. 准备该分量专属的工具
        path = os.path.join(self.args.checkpoints, setting['save_dir'], f'component_{component_idx}')
        if not os.path.exists(path):
            os.makedirs(path)
            
        time_now = time.time()
        train_steps = len(train_loader)
        
        # 专属 EarlyStopping
        early_stopping = EarlyStopping(patience=model_args.patience, verbose=True)
        
        # 专属 Optimizer (使用该分量的 model_args.learning_rate)
        model_optim = optim.Adam(model.parameters(), lr=model_args.learning_rate)
        
        # 专属 Loss
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        self.logger.info(f"Start training Component-{component_idx} | Epochs: {model_args.train_epochs} | LR: {model_args.learning_rate}")

        # =========================
        # Epoch Loop
        # =========================
        for epoch in range(model_args.train_epochs):
            iter_count = 0
            train_loss = []
            
            model.train()
            epoch_time = time.time()
            
            for iter_step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # ============================================================
                # 数据切分: 只取第 k+1 个分量 (跳过原始信号)
                # ============================================================
                inp = batch_x[:, :, :, component_idx+1]
                target = batch_y[:, :, :, component_idx+1]
                
                # Decoder Input
                dec_inp = torch.zeros_like(target[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([target[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward
                moe_loss = 0.0
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple): outputs, moe_loss = outputs[0], outputs[1]

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        target = target[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, target) + moe_loss * self.args.moe_weight
                        train_loss.append(loss.item())
                else:
                    outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs, moe_loss = outputs[0], outputs[1]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    target = target[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, target) + moe_loss * self.args.moe_weight
                    train_loss.append(loss.item())

                # Backward
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # End of Epoch
            self.logger.info(f"[Comp-{component_idx}] Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            
            # Validation (传入 component_idx 以便 vali 函数知道切分哪个数据)
            vali_loss = self._vali_single_component(model, vali_loader, criterion, component_idx)
            test_loss = self._vali_single_component(model, test_loader, criterion, component_idx) 

            self.logger.info(f"[Comp-{component_idx}] Epoch: {epoch + 1}, "
                             f"Steps: {train_steps} | "
                             f"Train Loss: {train_loss:.7f}, "
                             f"Vali Loss: {vali_loss:.7f}, "
                             f"Test Loss: {test_loss:.7f}")
            
            # Early Stopping
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                self.logger.info(f"[Comp-{component_idx}] Early stopping at epoch {epoch + 1}")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, model_args)

        # 加载该分量的最优模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        model.load_state_dict(torch.load(best_model_path))
        return model

    def _vali_single_component(self, model, loader, criterion, component_idx):
        """
        辅助函数：只验证单个分量
        """
        model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                inp = batch_x[:, :, :, component_idx+1]
                target = batch_y[:, :, :, component_idx+1]

                dec_inp = torch.zeros_like(target[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([target[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                
                if isinstance(outputs, tuple): outputs = outputs[0]
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = target[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, target)
                total_loss.append(loss.item())
        
        model.train()
        return np.average(total_loss)
    
    # ============================================================
    # 核心修改：train 函数支持指定 component
    # ============================================================
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        use_parallel = getattr(self.args, 'use_parallel', False) 
        
        # ----------------------------------------------------
        # [新增] 确定要训练的分量列表
        # ----------------------------------------------------
        if self.args.train_component is not None:
            # 确保输入合法
            if 0 <= self.args.train_component < self.args.num_imf:
                target_components = [self.args.train_component]
                self.logger.info(f">>> ONLY Training Component-{self.args.train_component} <<<")
            else:
                raise ValueError(f"train_component {self.args.train_component} is out of range [0, {self.args.num_imf-1}]")
        else:
            # 默认训练所有分量
            target_components = range(self.args.num_imf)
            self.logger.info(f">>> Training ALL Components: {list(target_components)} <<<")
        # ----------------------------------------------------

        if use_parallel:
            self.logger.info(">>> Starting Parallel Training <<<")
            
            # 只有当需要训练多个分量时，多进程才有意义；如果是单分量，直接串行即可
            if len(target_components) > 1:
                try:
                    mp.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass

                processes = []
                for k in target_components: # 遍历目标列表
                    model = self.model[k]
                    model_args = self.args.model_args_list[k]
                    
                    p = mp.Process(target=self._train_wrapper, args=(
                        k, model, model_args, setting, 
                        train_loader, vali_loader, test_loader
                    ))
                    p.start()
                    processes.append(p)
                
                for p in processes:
                    p.join()
                
                self.logger.info(">>> Parallel Training Finished. Reloading models... <<<")
                for k in target_components:
                    path = os.path.join(self.args.checkpoints, setting['save_dir'], f'component_{k}', 'checkpoint.pth')
                    if os.path.exists(path):
                        self.model[k].load_state_dict(torch.load(path))
            else:
                # 虽然开了 parallel 但只训练一个分量，直接退化为串行
                k = target_components[0]
                self._train_single_component_wrapper(k, setting, train_loader, vali_loader, test_loader)
                
        else:
            self.logger.info(">>>>> Starting Sequential Training <<<")
            for k in target_components: # 遍历目标列表
                self._train_single_component_wrapper(k, setting, train_loader, vali_loader, test_loader)

        return self.model

    def _train_single_component_wrapper(self, k, setting, train_loader, vali_loader, test_loader):
        """辅助函数，用于串行训练时的调用"""
        model_args = self.args.model_args_list[k]
        self.logger.info(f"--- Training Component {k} ---")
        
        trained_model = self._train_single_component(
            self.model[k], 
            model_args, 
            k, 
            setting,
            train_loader, 
            vali_loader, 
            test_loader
        )
        self.model[k] = trained_model
        
    # 多进程所需的 Wrapper (因为类方法直接传给 Process 容易 pickling error)
    def _train_wrapper(self, k, model, model_args, setting, train_loader, vali_loader, test_loader):
        # 重新设置一下 device，因为在子进程中
        device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.device = device # 更新子进程中的 self.device
        
        self._train_single_component(model, model_args, k, setting, train_loader, vali_loader, test_loader)
        
    def test(self, setting, test=0):
        # [未修改] test 函数不需要改动，它会自动检测 checkpoint 是否存在
        # 如果你只训练了 component-0，test 时会加载 comp-0，如果 comp-1 没训练，
        # 它会打印 warning 并跳过加载（使用随机初始化的参数），或者你可以根据需要修改 test 逻辑
        test_data, test_loader = self._get_data(flag='test')
        base_path = os.path.join(self.args.checkpoints, setting['save_dir'])

        self.logger.info('>>> Loading checkpoints for independent testing... <<<')
        for k in range(self.args.num_imf):
            comp_ckpt_path = os.path.join(base_path, f'component_{k}', 'checkpoint.pth')
            if os.path.exists(comp_ckpt_path):
                self.model[k].load_state_dict(torch.load(comp_ckpt_path))
                self.logger.info(f"Loaded Component-{k} from {comp_ckpt_path}")
            else:
                self.logger.warning(f"Checkpoint for Component-{k} not found at {comp_ckpt_path}! Using random init.")

        # 1. 初始化容器
        # 用于存储最终总和 (Original Scale)
        preds_total = []
        trues_total = []
        
        # [新增] 用于存储每个分量 (Normalized Scale)
        # 结构: [Comp0_List, Comp1_List, Comp2_List]
        preds_comps = [[] for _ in range(self.args.num_imf)]
        trues_comps = [[] for _ in range(self.args.num_imf)]
        
        result_path = os.path.join(self.args.results, setting['save_dir'])
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for model in self.model:
            model.eval()
            
        inference_time = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 初始化当前 Batch 的累加预测结果
                batch_pred_sum = 0
                
                start_time = time.time()

                # ============================================================
                # [核心逻辑] 遍历 K 个模型
                # ============================================================
                for k in range(self.args.num_imf):
                    model = self.model[k]
                    
                    # Input: component_k
                    inp = batch_x[:, :, :, k+1]
                    # Target: component_k (用于计算分量 Metrics)
                    target_k = batch_y[:, :, :, k+1]

                    # Decoder Input
                    dec_inp = torch.zeros_like(target_k[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([target_k[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                    
                    if isinstance(outputs, tuple): outputs = outputs[0]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] # [B, L, C]
                    
                    # ---------------------------------------------------
                    # [新增] 收集分量级结果 (保持在 GPU/Numpy 转换前)
                    # ---------------------------------------------------
                    # 注意：这里我们收集的是 Normalized 的数据，因为 scaler 不能用于分量
                    p_comp = outputs.detach().cpu().numpy()
                    t_comp = target_k[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                    
                    preds_comps[k].append(p_comp)
                    trues_comps[k].append(t_comp)
                    # ---------------------------------------------------

                    # 累加到总结果
                    batch_pred_sum += outputs

                inference_time += time.time() - start_time

                # ============================================================
                # [后处理] 总和结果 (Total Sum)
                # ============================================================
                batch_y_original = batch_y[:, :, :, 0]
                batch_y_original = batch_y_original[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = batch_pred_sum.detach().cpu().numpy()
                true = batch_y_original.detach().cpu().numpy()

                # 反归一化 (仅针对总和结果)
                if test_data.scale and self.args.inverse:
                    shape = pred.shape
                    pred = test_data.inverse_transform(pred.reshape(-1, shape[-1])).reshape(shape)
                    true = test_data.inverse_transform(true.reshape(-1, shape[-1])).reshape(shape)

                preds_total.append(pred)
                trues_total.append(true)

        # ============================================================
        # 1. 报告总和结果 (Original Scale)
        # ============================================================
        preds_total = np.concatenate(preds_total, axis=0)
        trues_total = np.concatenate(trues_total, axis=0)
        
        self.logger.info(f'Total Test Shape: {preds_total.shape}')
        avg_latency = (inference_time / (i + 1)) * 1000
        self.logger.info("Average Inference Latency: {:.2f} ms/batch".format(avg_latency))

        mae, mse, rmse, mape, mspe = metric(preds_total, trues_total)
        

        # ============================================================
        # 2. [新增] 报告每个分量的结果 (Normalized Scale)
        # ============================================================
        self.logger.info(f'>>>> COMPONENT Metrics (Normalized Scale) <<<<')
        comp_metrics = []
        
        for k in range(self.args.num_imf):
            # 拼接该分量的所有 batch
            p_k = np.concatenate(preds_comps[k], axis=0)
            t_k = np.concatenate(trues_comps[k], axis=0)
            mae_k, mse_k, rmse_k, _, _ = metric(p_k, t_k)
            self.logger.info(f'Component-{k}: mse:{mse_k:.5f}, mae:{mae_k:.5f}')
            comp_metrics.append([mse_k, mae_k])
            
        self.logger.info(f'>>>> TOTAL SUM Metrics (Original Scale) <<<<')
        self.logger.info('mse:{:.5f}, mae:{:.5f}, rmse:{:.5f}'.format(mse, mae, rmse))
        # ============================================================
        # 保存
        # ============================================================
        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_path, 'pred.npy'), preds_total)
        np.save(os.path.join(result_path, 'true.npy'), trues_total)
        
        # 可选：保存分量的 Metrics
        np.save(os.path.join(result_path, 'metrics_components.npy'), np.array(comp_metrics))
        self.logger.info(f">>>>>>>>>>>>>>>>>>>>>> saved metrics to {result_path}\n\n")
        return
    
    