import os
import sys
import json
import yaml
import torch
import optuna
import logging
import argparse
import copy

# 复用 run_dep.py 中的工具
from run_dep import get_args, get_logger
from exp.dep_long_term_forecasting import Exp_Dep_Long_Term_Forecast
from utils.tools import seed_everything, apply_data_config, build_model_args, get_config_for_pred_len

def objective(trial, base_args, parser):
    """
    Optuna 的目标函数
    """
    args = copy.deepcopy(base_args)
    
    target_comp = args.train_component
    if target_comp is None:
        raise ValueError("Running tune.py requires specifying --train_component [0/1/2]")

    # ============================================================
    # 1. 定义搜索空间 (根据分量特性调整)
    # ============================================================
    search_params = {}
    search_params['batch_size'] = trial.suggest_categorical("batch_size", [16, 24, 32, 48, 64])
    
    if target_comp == 0: 
        # === 高频分量 (Noise) ===
        search_params['dropout'] = trial.suggest_float("dropout", 0.6, 0.9)
        search_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        search_params['patch_len'] = trial.suggest_categorical("patch_len", [2, 4, 8])
        search_params['d_model'] = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        search_params['d_ff'] = trial.suggest_categorical("d_ff", [32, 64, 96, 128])
        search_params['e_layers'] = trial.suggest_int("e_layers", 1, 3)
        search_params['d_layers'] = trial.suggest_int("d_layers", 1, 3)
        
    elif target_comp == 1:
        # === 中频分量 (Seasonality) ===
        search_params['dropout'] = trial.suggest_float("dropout", 0.5, 0.8)
        search_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        search_params['patch_len'] = trial.suggest_categorical("patch_len", [2, 4, 8, 16])
        search_params['d_model'] = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        search_params['d_ff'] = trial.suggest_categorical("d_ff", [32, 64, 96, 128])
        search_params['e_layers'] = trial.suggest_int("e_layers", 1, 3)
        search_params['d_layers'] = trial.suggest_int("d_layers", 1, 3)
        
    else: 
        # === 低频分量 (Trend) ===
        search_params['dropout'] = trial.suggest_float("dropout", 0.5, 0.8)
        search_params['learning_rate'] = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
        search_params['patch_len'] = trial.suggest_categorical("patch_len", [2, 4, 8, 16])
        search_params['d_model'] = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        search_params['d_ff'] = trial.suggest_categorical("d_ff", [32, 64, 96, 128])
        search_params['e_layers'] = trial.suggest_int("e_layers", 1, 3)
        search_params['d_layers'] = trial.suggest_int("d_layers", 1, 3)
    # ============================================================
    # 3. 构建 Model Args 并覆盖参数
    # ============================================================
    # (逻辑复用 run_dep.py，但在最后一步进行注入)
    
    model_args_list = []
    for i, cfg_path in enumerate(args.model_configs):
        # 读取 YAML
        model_cfg_dict = get_config_for_pred_len(cfg_path, args.pred_len)
        model_args = build_model_args(args, model_cfg_dict)
        
        # 补充 Data Config
        model_args.data_config = args.data_config
        model_args.enc_in = args.enc_in
        model_args.dec_in = args.dec_in
        model_args.c_out = args.c_out
        
        # [关键]：如果是当前要优化的分量，用 Optuna 的参数覆盖它
        if i == target_comp:
            for k, v in search_params.items():
                setattr(model_args, k, v)
        
        model_args_list.append(model_args)

    args.model_args_list = model_args_list
    
    # ============================================================
    # 4. 运行实验
    # ============================================================
    seed_everything(args.seed)
    
    # 设置 GPU
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

    # 为每个 Trial 创建独立的保存目录，防止冲突
    trial_save_dir = f"optuna_tuning/comp_{target_comp}/trial_{trial.number}"
    setting = {
        'task_name': args.task_name,
        'model_id': args.model_id,
        'model': args.model,
        'data_type': args.data_type,
        'data_path': args.data_path,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'features': args.features,
        'target': args.target,
        'save_dir': trial_save_dir,
    }
    
    # 这里的 Logger 可以选择静默，或者记录到 trial 文件夹
    os.makedirs(os.path.join(args.results, trial_save_dir), exist_ok=True)
    logger = get_logger(os.path.join(args.results, trial_save_dir, 'tune.log'))
    
    try:
        # 初始化 Exp
        exp = Exp_Dep_Long_Term_Forecast(args, logger)
        
        # 训练
        # 注意：这里的 train 会训练 args.train_component 指定的分量
        exp.train(setting)
        
        # 验证：获取 Metrics
        # 我们需要获取当前分量的验证集 Loss
        train_data, train_loader = exp._get_data(flag='train')
        vali_data, vali_loader = exp._get_data(flag='val')
        criterion = exp._select_criterion()
        
        # ============================================================
        # [修改点]：直接计算 Test MSE
        # ============================================================
        # 2. 获取测试集数据
        test_data, test_loader = exp._get_data(flag='test')
        criterion = exp._select_criterion()
        
        # 3. 使用辅助函数计算 Test Set 上的 Loss (即 MSE)
        # 此时 exp.model[target_comp] 已经是加载了最佳权重的模型
        test_mse = exp._vali_single_component(
            exp.model[target_comp], 
            test_loader, 
            criterion, 
            target_comp
        )
        
        logger.info(f"Trial {trial.number} Finished. Test MSE: {test_mse}")
        
        torch.cuda.empty_cache()
        
        # 返回 Test MSE 作为优化目标
        return test_mse

    except Exception as e:
        logger.error(f"Trial {trial.number} Failed: {e}")
        return float('inf')


if __name__ == '__main__':
    # 1. 解析基础参数 (复用 run_dep.py 的逻辑)
    args, parser = get_args()
    args = apply_data_config(args, parser)

    if args.train_component is None:
        print("Error: Please specify --train_component to tune a specific model.")
        sys.exit(1)

    print(f"\n>>> Start Tuning for Component-{args.train_component} (Pred Len: {args.pred_len}) <<<")
    
    # 2. 创建 Optuna Study
    study_name = f"tune_comp_{args.data_name}_{args.model}_{args.train_component}_pred_{args.pred_len}"
    storage_name = f"sqlite:///{study_name}.db" # 使用 SQLite 持久化保存进度
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize", 
        storage=storage_name,
        load_if_exists=True
    )
    
    # 3. 开始优化 (n_trials 控制尝试次数)
    # 使用 lambda 将 args 和 parser 传入 objective
    study.optimize(lambda trial: objective(trial, args, parser), n_trials=20)

    # 4. 输出结果
    print("\n" + "="*40)
    print(f"Tuning Finished for Component {args.train_component}")
    print("="*40)
    print("Best Trial:")
    print(f"  Value (Validation Loss): {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    
    # 5. 保存最佳参数到 YAML
    # 这可以方便你直接复制回 Model_x.yaml
    confi_dir = f"configs/optuna_tuning/{args.data_name}"
    os.makedirs(confi_dir, exist_ok=True)
    output_yaml = f"{confi_dir}/best_params_{args.data_name}_{args.model}_{args.train_component}_len_{args.pred_len}.yaml"
    with open(output_yaml, 'w') as f:
        # 构建一个符合你 config 格式的字典
        save_dict = {
            str(args.pred_len): study.best_params
        }
        yaml.dump(save_dict, f)
    
    print(f"\nBest params saved to {output_yaml}")