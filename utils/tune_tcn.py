import os
import sys
import json
import yaml
import torch
import optuna
import logging
import argparse
import copy
import traceback # 引入 traceback 以便打印详细错误

# 复用 run_dep.py 中的工具
from run_dep import get_args, get_logger
from exp.dep_long_term_forecasting import Exp_Dep_Long_Term_Forecast # 注意引用路径
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
        search_params['dropout'] = trial.suggest_float("dropout", 0.3, 0.8)
        search_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        search_params['kernel_size'] = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        search_params['d_model'] = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        search_params['e_layers'] = trial.suggest_int("e_layers", 1, 3)
        search_params['d_layers'] = trial.suggest_int("d_layers", 1, 3)
        
    elif target_comp == 1:
        # === 中频分量 (Seasonality) ===
        search_params['dropout'] = trial.suggest_float("dropout", 0.3, 0.8)
        search_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        search_params['kernel_size'] = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        search_params['d_model'] = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        search_params['e_layers'] = trial.suggest_int("e_layers", 1, 3)
        search_params['d_layers'] = trial.suggest_int("d_layers", 1, 3)
        
    else: 
        # === 低频分量 (Trend) ===
        search_params['dropout'] = trial.suggest_float("dropout", 0.3, 0.8)
        search_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        search_params['kernel_size'] = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        search_params['d_model'] = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        search_params['e_layers'] = trial.suggest_int("e_layers", 1, 3)
        search_params['d_layers'] = trial.suggest_int("d_layers", 1, 3)

    # ============================================================
    # 2. 注入参数
    # ============================================================
    model_args_list = []
    for i, cfg_path in enumerate(args.model_configs):
        model_cfg_dict = get_config_for_pred_len(cfg_path, args.pred_len)
        model_args = build_model_args(args, model_cfg_dict)
        
        model_args.data_config = args.data_config
        model_args.enc_in = args.enc_in
        model_args.dec_in = args.dec_in
        model_args.c_out = args.c_out
        
        if i == target_comp:
            for k, v in search_params.items():
                setattr(model_args, k, v)
        
        model_args_list.append(model_args)

    args.model_args_list = model_args_list
    
    # ============================================================
    # 3. 运行实验
    # ============================================================
    seed_everything(args.seed)
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

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
    
    os.makedirs(os.path.join(args.results, trial_save_dir), exist_ok=True)
    logger = get_logger(os.path.join(args.results, trial_save_dir, 'tune.log'))
    
    try:
        exp = Exp_Dep_Long_Term_Forecast(args, logger)
        
        # 1. 训练 (Train)
        # 此时 exp.train 会训练模型，保存 Checkpoint，然后销毁内存中的模型
        exp.train(setting)
        
        # ============================================================
        # [核心修复]：手动重建并加载模型
        # ============================================================
        
        # A. 重建模型结构
        # 使用 exp 中提供的辅助函数 _build_individual_model
        model = exp._build_individual_model(target_comp)
        
        # B. 加载刚才训练保存的 Checkpoint
        ckpt_path = os.path.join(args.checkpoints, setting['save_dir'], f'component_{target_comp}', 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            
        model.load_state_dict(torch.load(ckpt_path))
        
        # 2. 获取测试集数据
        test_data, test_loader = exp._get_data(flag='test')
        criterion = exp._select_criterion()
        
        # 3. 验证 (计算 Test MSE)
        test_mse = exp._vali_single_component(
            model,  # 传入刚加载的模型
            test_loader, 
            criterion, 
            target_comp
        )
        
        logger.info(f"Trial {trial.number} Finished. Test MSE: {test_mse}")
        
        # 4. 再次清理显存
        del model
        torch.cuda.empty_cache()
        
        return test_mse

    except Exception as e:
        logger.error(f"Trial {trial.number} Failed: {e}")
        logger.error(traceback.format_exc()) # 打印完整堆栈信息
        return float('inf')


if __name__ == '__main__':
    args, parser = get_args()
    args = apply_data_config(args, parser)

    if args.train_component is None:
        print("Error: Please specify --train_component to tune a specific model.")
        sys.exit(1)

    print(f"\n>>> Start Tuning for Component-{args.train_component} (Pred Len: {args.pred_len}) <<<")
    
    study_name = f"tune_comp_{args.data_name}_{args.model}_{args.train_component}_pred_{args.pred_len}"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize", 
        storage=storage_name,
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, args, parser), n_trials=20)

    print("\n" + "="*40)
    print(f"Tuning Finished for Component {args.train_component}")
    print("="*40)
    print("Best Trial:")
    print(f"  Value (Test MSE): {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    
    confi_dir = f"configs/optuna_tuning/{args.data_name}"
    os.makedirs(confi_dir, exist_ok=True)
    output_yaml = f"{confi_dir}/best_params_{args.data_name}_{args.model}_{args.train_component}_len_{args.pred_len}.yaml"
    with open(output_yaml, 'w') as f:
        save_dict = {
            str(args.pred_len): study.best_params
        }
        yaml.dump(save_dict, f)
    
    print(f"\nBest params saved to {output_yaml}")
    