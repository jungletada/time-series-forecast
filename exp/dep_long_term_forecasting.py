import os
import time
import warnings
import numpy as np
import json
import torch
import torch.nn as nn
from torch import optim
import torch.multiprocessing as mp
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dtw_metric import dtw, accelerated_dtw
from utils.loss import WeightedL1Loss

warnings.filterwarnings('ignore')

class Exp_Dep_Long_Term_Forecast(Exp_Basic):
    """
     train and test the NDA-based long term forecasting model.
    """
    
    def __init__(self, args, logger):
        self.logger = logger
        self.logger.info(f'Initializing Exp_Long_Term_Forecast (Training {args.num_imf} Independent Models).')
        self.logger.info(f'Number of components: {args.num_imf}')
        super(Exp_Dep_Long_Term_Forecast, self).__init__(args)
        self.model = None 

    def _build_model(self):
        # 这个函数在父类 Exp_Basic 中可能会被调用
        # 但我们现在的策略是按需构建，所以这里返回 None 或者报错，或者留空
        return None

    def _build_individual_model(self, component_idx):
        """
        [新增函数]：只构建指定 index 的这一个模型
        """
        model_args = self.args.model_args_list[component_idx]
        self.logger.info(f'>>> Building INDIVIDUAL model for Component-{component_idx} with args: {model_args}')
        
        model = self.model_dict[model_args.model].Model(model_args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # 必须手动搬运到 GPU，因为不再由 Exp_Basic 统一管理
        if self.args.use_gpu:
            model = model.to(self.device)
            
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        # print(f">>>>>>>>>>>>>>>>>>>>>> length of data_set: {len(data_set)}")
        return data_set, data_loader

    def _select_optimizer(self, model, component_idx):
        """
        [修改]：只为当前传入的单个模型创建优化器
        """
        model_args = self.args.model_args_list[component_idx]
        lr = model_args.learning_rate
        if self.args.optimizer == 'AdamW':
            model_optim = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=0.01,
                betas=(0.9, 0.999),   
                eps=1e-8                     
            )
        elif self.args.optimizer == 'Adam':
            model_optim = optim.Adam(
                model.parameters(), 
                lr=lr)
            
        elif self.args.optimizer == 'SGD':
            model_optim = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=0.01                     
            )
        elif self.args.optimizer == 'RMSprop':
            model_optim = optim.RMSprop(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=0.01                     
            )
        elif self.args.optimizer == 'Adagrad':
            model_optim = optim.Adagrad(
                model.parameters(),
                lr=lr,
                weight_decay=0.01                     
            )
        elif self.args.optimizer == 'Adadelta':
            model_optim = optim.Adadelta(
                model.parameters(),
                lr=lr,
                weight_decay=0.01                     
            )
        elif self.args.optimizer == 'LBFGS':
            model_optim = optim.LBFGS(
                model.parameters(),
                lr=lr,
                weight_decay=0.01                     
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optimizer}")

        self.logger.info(f"Built Optimizer for Component-{component_idx} with lr={lr}")
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'L1':
            criterion = nn.L1Loss()
        elif self.args.loss == 'WeightedL1':
            criterion = WeightedL1Loss(self.args.lossfun_alpha, self.args.loss_mode)
        else:
            raise ValueError(f"Invalid loss: {self.args.loss}")
        return criterion
    
    def _train_single_component(self, model, model_args, component_idx, setting, train_loader, vali_loader, test_loader):
        """
        独立训练单个分量的函数 (逻辑基本不变，只是 model 从参数传入)
        """
        path = os.path.join(self.args.checkpoints, setting['save_dir'], f'component_{component_idx}')
        if not os.path.exists(path):
            os.makedirs(path)
            
        time_now = time.time()
        train_steps = len(train_loader)
        
        early_stopping = EarlyStopping(patience=model_args.patience, verbose=True)
        
        # [修改] 使用新的优化器生成函数
        model_optim = self._select_optimizer(model, component_idx)
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        self.logger.info(f"Start training Component-{component_idx} | Epochs: {model_args.train_epochs}")

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
                
                inp = batch_x[:, :, :, component_idx+1]
                target = batch_y[:, :, :, component_idx+1]
                
                dec_inp = torch.zeros_like(target[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([target[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple): outputs = outputs[0]
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        target = target[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, target)
                        train_loss.append(loss.item())
                else:
                    outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    target = target[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, target)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            self.logger.info(f"[Comp-{component_idx}] Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            
            vali_loss = self._vali_single_component(model, vali_loader, criterion, component_idx)
            test_loss = self._vali_single_component(model, test_loader, criterion, component_idx) 

            self.logger.info(f"[Comp-{component_idx}] Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
            
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                self.logger.info(f"[Comp-{component_idx}] Early stopping at epoch {epoch + 1}")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, model_args)

        # 加载最优模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        model.load_state_dict(torch.load(best_model_path))
        return model

    def _vali_single_component(self, model, loader, criterion, component_idx):
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
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # 确定要训练的分量列表
        if self.args.train_component is not None:
            if 0 <= self.args.train_component < self.args.num_imf:
                target_components = [self.args.train_component]
                self.logger.info(f">>> ONLY Training Component-{self.args.train_component} <<<")
            else:
                raise ValueError(f"train_component {self.args.train_component} is out of range")
        else:
            target_components = range(self.args.num_imf)
            self.logger.info(f">>> Training ALL Components: {list(target_components)} <<<")

        # 注意：为了解决显存问题，我们暂时只支持串行训练
        # 如果使用 parallel，所有进程同时启动，显存依然会爆
        # 下面是串行逻辑：
        
        self.logger.info(">>>>> Starting Sequential Training (Memory Optimized) <<<")
        for k in target_components: 
            self.logger.info(f"--- Training Component {k} ---")
            model_args = self.args.model_args_list[k]
            
            # [关键步骤 1] 实例化单个模型
            current_model = self._build_individual_model(k)
            
            # [关键步骤 2] 训练
            trained_model = self._train_single_component(
                current_model, 
                model_args, 
                k, 
                setting,
                train_loader, 
                vali_loader, 
                test_loader
            )
            
            # [关键步骤 3] 训练完立刻释放资源
            self.logger.info(f"--- Finished Component {k}. Releasing memory... ---")
            del current_model
            del trained_model
            torch.cuda.empty_cache() # 强制释放显存
            
        return None # 不再返回 self.model 列表，因为已经被销毁了

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        base_path = os.path.join(self.args.checkpoints, setting['save_dir'])
        result_path = os.path.join(self.args.results, setting['save_dir'])
        visual_path = os.path.join(result_path, 'visual')
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # ----------------------------------------------------
        # 内存优化版测试逻辑
        # ----------------------------------------------------
        # 我们不能同时加载所有模型。
        # 策略：依次加载模型 -> 预测全量测试集 -> 存入CPU列表 -> 释放显存 -> 加载下一个
        
        # 存储所有分量的预测结果 (在 CPU 上)
        # preds_comps_all: List[Array(N, Pred_Len, C)]
        preds_comps_all = []
        trues_comps_all = []
        
        # 加载所有需要的 Component (或者只测试训练的那一个，取决于 args.train_component)
        # 这里假设测试阶段我们需要所有分量来合成最终结果
        
        self.logger.info('>>> Starting Memory-Optimized Testing... <<<')
        
        for k in range(self.args.num_imf):
            self.logger.info(f">>> Testing Component-{k} <<<")
            
            # 1. 检查 Checkpoint
            comp_ckpt_path = os.path.join(base_path, f'component_{k}', 'checkpoint.pth')
            if not os.path.exists(comp_ckpt_path):
                self.logger.warning(f"Checkpoint for Component-{k} not found! Skipping logic requires attention.")
                # 这里如果缺失模型，可能需要用全0填充，或者报错
                # 简单起见，这里假设必然存在，或者用随机初始化模型（但不 load 权重）
            
            # 2. 构建并加载模型
            model = self._build_individual_model(k)
            if os.path.exists(comp_ckpt_path):
                model.load_state_dict(torch.load(comp_ckpt_path))
                self.logger.info(f"Loaded weights for Component-{k}")
            model.eval()
            
            # 3. 预测当前分量的全量测试集
            preds_k = []
            trues_k = []
            
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    inp = batch_x[:, :, :, k+1]
                    target_k = batch_y[:, :, :, k+1]

                    dec_inp = torch.zeros_like(target_k[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([target_k[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = model(inp, batch_x_mark, dec_inp, batch_y_mark)
                    
                    if isinstance(outputs, tuple): outputs = outputs[0]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    
                    # 收集结果 (立即转 CPU)
                    preds_k.append(outputs.detach().cpu().numpy())
                    trues_k.append(target_k[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())
            
            # 4. 释放显存
            del model
            torch.cuda.empty_cache()
            
            # 整理当前分量结果
            preds_k = np.concatenate(preds_k, axis=0) # [N, Pred, C]
            trues_k = np.concatenate(trues_k, axis=0)
            
            preds_comps_all.append(preds_k)
            trues_comps_all.append(trues_k)
            
            self.logger.info(f"Component-{k} Inference Done. Memory Released.")

        # ============================================================
        # 结果汇总与评估
        # ============================================================
        
        # 1. 计算每个分量的 Metrics
        self.logger.info(f'>>>> COMPONENT Metrics (Normalized Scale) <<<<')
        comp_metrics = []
        for k in range(self.args.num_imf):
            p_k = preds_comps_all[k]
            t_k = trues_comps_all[k]
            mae_k, mse_k, rmse_k, _, _ = metric(p_k, t_k)
            self.logger.info(f'Component-{k}: mse:{mse_k:.5f}, mae:{mae_k:.5f}')
            comp_metrics.append([mse_k, mae_k])

        # 2. 计算总和 (Total Sum)
        # 将 list of arrays stack 起来 -> [K, N, Pred, C] -> sum axis 0 -> [N, Pred, C]
        preds_sum = np.sum(np.stack(preds_comps_all), axis=0)
        
        # 严谨的做法是重新读一次 dataset：原始 target 与原始输入 batch_x[:, :, :, 0]（与 preds_sum 顺序一致）
        self.logger.info(">>>>>>> Retrieving Original Input / Ground Truth... <<<")
        trues_original = []
        inputs_original = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                inp_orig = batch_x[:, :, :, 0]
                orig = batch_y[:, :, :, 0]
                orig = orig[:, -self.args.pred_len:, f_dim:]
                inputs_original.append(inp_orig.detach().cpu().numpy())
                trues_original.append(orig.detach().cpu().numpy())

        trues_total = np.concatenate(trues_original, axis=0)
        inputs_total = np.concatenate(inputs_original, axis=0)

        # 3. 反归一化
        if test_data.scale and self.args.inverse:
            shape = preds_sum.shape
            preds_total = test_data.inverse_transform(preds_sum.reshape(-1, shape[-1])).reshape(shape)
            trues_total = test_data.inverse_transform(
                trues_total.reshape(-1, trues_total.shape[-1])).reshape(trues_total.shape)
            shape_in = inputs_total.shape
            inputs_total = test_data.inverse_transform(
                inputs_total.reshape(-1, shape_in[-1])).reshape(shape_in)
        else:
            preds_total = preds_sum

        if self.args.visualize == 1:
            offset = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                bs = batch_x.shape[0]
                if i % 2 == 0:
                    idx = offset
                    input_np = inputs_total[idx]
                    horizon_len = input_np.shape[0]
                    hist = input_np[:, -1]
                    true_f = trues_total[idx, :, -1]
                    pred_f = preds_total[idx, :, -1]
                    label = np.concatenate((hist, true_f), axis=0)
                    prediction = np.concatenate((hist, pred_f), axis=0)
                    ## 计算 true_f 和 pred_f 的均方误差（MSE）
                    # mse_cur = np.mean((true_f - pred_f) ** 2)
                    # self.logger.info(f"Sample-{i}: MSE between true and pred: {mse_cur:.6f}")
                    pdf_save_path = os.path.join(visual_path, str(i) + '.pdf')
                    visual(
                        label,
                        prediction,
                        horizon_len,
                        pdf_save_path,
                        title=self.args.model_id)
                    # self.logger.info(f"Saved visualization to {pdf_save_path}")
                offset += bs

            true_full = trues_total[:, :, -1].reshape(-1)
            pred_full = preds_total[:, :, -1].reshape(-1)
            full_pdf = os.path.join(visual_path, 'full_pred_vs_true.pdf')
            visual(
                true_full,
                pred_full,
                None,
                full_pdf,
                title=self.args.model_id,
                figsize=(48, 6))
            # self.logger.info(f"Saved full pred vs true comparison to {full_pdf}")

        self.logger.info(f'Total Test Shape: {preds_total.shape}')
        mae, mse, rmse, mape, mspe = metric(preds_total, trues_total)
        self.logger.info(f'>>>> TOTAL SUM Metrics (Original Scale) <<<<')
        self.logger.info('mse:{:.5f}, mae:{:.5f}, rmse:{:.5f}'.format(mse, mae, rmse))

        # Save results as numpy arrays
        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_path, 'pred.npy'), preds_total)
        np.save(os.path.join(result_path, 'true.npy'), trues_total)
        np.save(os.path.join(result_path, 'metrics_components.npy'), np.array(comp_metrics))
        self.logger.info(f">>>>>>>>>>>>>>>>>>>>>> saved metrics to {result_path}\n\n")
        
        return
    