import os
import sys
import yaml
import logging
import argparse
import json
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from utils.tools import seed_everything, apply_data_config, apply_model_config

def get_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    # basic config
    parser.add_argument('--seed', type=int, default=2026, help='random seed')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]',
                        choices=['long_term_forecast', 'short_term_forecast', 'imputation', 'classification', 'anomaly_detection'])
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--use_mnn', type=int, default=0, help='use mnn for inference.')
    parser.add_argument('--mnn', type=str, default='mlp', help='mnn model name, options: [mlp, tcn, wpmixer]')
    parser.add_argument('--num_imf', type=int, default=3, help='number of imfs')
    parser.add_argument('--visualize', type=int, default=1, help='visualize')
    # data loader
    parser.add_argument('--data_config', type=str, default='configs/datasets/dataset.yaml', help='data config')
    parser.add_argument('--model_config', type=str, default=None, help='model config')
    parser.add_argument('--decomp_k', type=int, default=3, help='decomposition k')
    parser.add_argument('--data_name', type=str, default='ETTh1', help='dataset name')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; " \
                        "M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate',
                        choices=['M', 'S', 'MS'])
    parser.add_argument('--target', default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily," \
                            "b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./results/', help='location of result files')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--rnn_type', type=str, default='RNN', help='RNN type, options: [RNN, LSTM, GRU]')
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--nda_patch', type=int, default=4, help='patch length for NDA')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--kernel_size', type=int, default=5, help='kernel size for TCN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--moe_weight', type=float, default=0.05, help='moe loss weight')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    # parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # 1. 先按代码默认值 + 命令行解析一遍
    args = parser.parse_args()

    # 2. 用 model_config.yaml 覆盖还在默认值的模型参数（命令行优先）
    args = apply_model_config(args, parser)

    # 3. 用 data_config.yaml 覆盖还在默认值的数据参数（命令行优先）
    args = apply_data_config(args, parser)

    return args

def get_logger(log_file='run.log'):
    """Set up logging to a file, replacing print with logger.info or logger.error
    """
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.INFO)
    # Create handlers
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add handlers to logger (avoid duplicate handlers on multiple calls)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

if __name__ == '__main__':
    args = get_args()
    
    seed_everything(args.seed)
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        Exp = Exp_Zero_Shot_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for exp_time in range(args.itr):
            # setting record of experiments
            if args.data_type == 'm4':
                save_path =f'{args.task_name}_{args.data_name}_{args.model}_#{exp_time}'
            else:
                save_path =f'{args.task_name}_{args.data_name}_{args.model}_seq{args.seq_len}_pred{args.pred_len}_ft({args.features})_#{exp_time}'
            save_dir = os.path.join(args.data_name, save_path)
            os.makedirs(os.path.join(args.results, save_dir), exist_ok=True)
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
                'exp_time': exp_time,
                'save_dir': save_dir,
            }
            logger = get_logger(os.path.join(args.results, save_dir, 'run.log'))
            logger.info(json.dumps(vars(args), default=str))
            exp = Exp(args, logger)  # set experiments
            logger.info(f'\n\n>>>>>>>start training : {json.dumps(setting)} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            logger.info(f'\n\n>>>>>>>testing : {json.dumps(setting)} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    
    else:
        exp_time = 0
        # setting record of experiments
        if args.data_type == 'm4':
            save_path =f'{args.task_name}_{args.data_name}_{args.model}_#{exp_time}'
        else:
            save_path =f'{args.task_name}_{args.data_name}_{args.model}_seq{args.seq_len}_pred{args.pred_len}_ft({args.features})_#{exp_time}'
        save_dir = os.path.join(args.data_name, save_path)
        os.makedirs(os.path.join(args.results, save_dir), exist_ok=True)
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
            'exp_time': exp_time,
            'save_dir': save_dir,
        }
        logger = get_logger(os.path.join(args.results, save_dir, 'test.log'))
        logger.info(json.dumps(vars(args), default=str))
        exp = Exp(args, logger)  # set experiments
        logger.info(f'\n\n>>>>>>>testing : {json.dumps(setting)} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
