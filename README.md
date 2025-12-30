# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

## Getting Started

### Prepare Data
You can obtain the well-preprocessed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library). Then place the downloaded data in the folder `dataset`.

### Installation
1. Clone this repository.
   ```bash
   git clone https://github.com/thuml/Time-Series-Library.git
   cd Time-Series-Library
   ```

2. Create a new Conda environment.
   ```bash
   conda create -n tslib python=3.11
   conda activate tslib
   ```

3. Install Core Dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Install Dependencies for Mamba Model (Required for Time-Series-Library/models/Mamba.py)
   > ⚠️ **CUDA Compatibility Notice**
   > The prebuilt Mamba wheel is **CUDA-version specific**.
   > Please make sure to install the wheel that matches your local CUDA version
   > (e.g., `cu11` or `cu12`). Installing a mismatched version may result in
   > runtime errors or import failures.

   Example for **CUDA 12**:

   ```bash
   pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   ```

5. Install Dependencies for Moirai Model (Required for Time-Series-Library/models/Moirai.py)
   ```bash
   pip install uni2ts --no-deps
   ```

### Docker Deployment
```bash
# Build and start the Docker container in detached mode
docker compose -f 'Time-Series-Library/docker-compose.yml' up -d --build

# Download / place the dataset into a newly created folder ./dataset at the repository root
mkdir -p dataset  # create the dataset directory

# Copy the local dataset into the container at /workspace/dataset
docker cp ./dataset tslib:/workspace/dataset

# Enter the running container to continue training / evaluation
docker exec -it tslib bash

# Switch to the workspace directory inside the container
cd /workspace

# Run zero-shot forecasting with the pre-trained Moirai model
python -u run.py \
  --task_name zero_shot_forecast \   # task type: zero-shot forecasting
  --is_training 0 \                  # 0 = inference only (no training)
  --root_path ./dataset/ETT-small/ \ # root directory of the dataset
  --data_path ETTh1.csv \            # dataset file name
  --model_id ETTh1_512_96 \          # experiment/model identifier
  --model Moirai \                   # model name (TimesFM / Moirai)
  --data ETTh1 \                     # dataset name
  --features M \                     # multivariate forecasting
  --seq_len 512 \                    # input sequence length
  --pred_len 96 \                    # prediction horizon
  --enc_in 7 \                       # number of input variables
  --des 'Exp' \                      # experiment description
  --itr 1                             # number of runs
```


### Quick Test

Quick test for all 5 tasks (1 epoch each):

```bash
# Run quick tests for all 5 tasks
export CUDA_VISIBLE_DEVICES=0
```
```bash
# 1. Long-term forecasting
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id test_long \
    --model DLinear \
    --features M \
    --seq_len 96 --pred_len 96 \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --train_epochs 1 \
    --num_workers 2
```

```bash
# 2. Short-term forecasting (using ETT dataset with shorter prediction length)
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id test_short \
  --model TimesNet \
  --features M \
  --seq_len 24 --label_len 12 --pred_len 24 \
  --e_layers 2 --d_layers 1 --d_model 16 \
  --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 5 \
  --train_epochs 1 \
  --num_workers 2
```

### Train and Evaluate
We provide the experiment scripts for all benchmarks under the folder `scripts/`. You can reproduce the experiment results as the following examples:

```bash
# long-term forecast
bash scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash scripts/short_term_forecast/TimesNet_M4.sh
```

### Develop Your Own Model
- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `scripts`.

### Inspect the project structure:

```
Time-Series-Library/
├── README.md                     # Official README with tasks, leaderboard, usage
├── requirements.txt              # pip dependency list for quick environment setup
├── LICENSE / CONTRIBUTING.md     # Upstream license and contribution guide
├── run.py                        # Unified entry that parses args and dispatches tasks
├── exp/                          # Task pipelines wrapping train/val/test
│   ├── exp_basic.py              # Experiment base class, registers models, builds flows
│   ├── exp_long_term_forecasting.py    # Long-term forecasting logic
│   ├── exp_short_term_forecasting.py   # Short-term forecasting logic
│   ├── exp_imputation.py               # Missing-value imputation
│   ├── exp_anomaly_detection.py        # Anomaly detection
│   ├── exp_classification.py           # Classification
│   └── exp_zero_shot_forecasting.py    # LTSM zero-shot evaluation
├── data_provider/                # Dataset loaders and splits
│   ├── data_factory.py           # Chooses the proper DataLoader per task
│   ├── data_loader.py            # Generic TS reader with sliding-window logic
│   ├── uea.py / m4.py            # Parsers for UEA, M4 and other formats
│   └── __init__.py               # Exposes factory interfaces upward
├── models/                       # All model implementations
│   ├── TimesNet.py, TimeMixer.py # Main forecasting models
│   ├── Chronos2.py, TiRex.py     # LTSM zero-shot models
│   └── __init__.py               # Enables name-based instantiation inside exp
├── layers/                       # Reusable attention / conv / embedding blocks
│   ├── Transformer_EncDec.py     # Transformer stacks
│   ├── AutoCorrelation.py        # Auto-correlation operator
│   ├── MultiWaveletCorrelation.py# Frequency-domain unit
│   └── Embed.py etc.             # Shared primitives
├── utils/                        # Utility toolbox
│   ├── metrics.py                # MSE / MAE / DTW and other metrics
│   ├── tools.py                  # General helpers such as EarlyStopping
│   ├── augmentation.py           # Augmentations for classification / detection
│   ├── print_args.py             # Unified argument printer
│   └── masking.py / losses.py    # Task-specific helpers
├── scripts/                      # Bash recipes for reproducible experiments
│   ├── long_term_forecast/       # Long-term forecasting per dataset/model
│   ├── short_term_forecast/      # M4 and other short-term scripts
│   ├── imputation/               # Imputation scripts
│   ├── anomaly_detection/        # SMD / SMAP / SWAT detection scripts
│   ├── classification/           # UEA classification scripts
│   └── exogenous_forecast/       # TimeXer exogenous forecasting flow
├── tutorial/                     # TimesNet tutorial notebook and figures
└── pic/                          # README figures (dataset overview, etc.)
```

```
时间  →   ...  t-96           t-48               t                              t+96  ...
              |----------------|----------------|--------------------------------|
                 seq_len = 96      label_len=48            pred_len=96

数据切片：

1) encoder 输入  batch_x  (长度 = seq_len)
              [ t-96  .............................................  t-1 ]

2) decoder 监督  batch_y  (长度 = label_len + pred_len)
              [ t-48  ..................  t-1 |  t  ................  t+95 ]
                ↑--------- label_len --------↑   ↑------ pred_len --------↑

3) decoder 实际输入`dec_inp`  (长度 = label_len + pred_len)
              [   真实值(teacher tokens) |         0 占位(未来未知)        ]
              [ t-48 .......... t-1     |   0  0  0  ...  0  (共 96 个)  ]

4) loss 计算区间（只对 pred_len 的未来算）
                                        [ t  ....................  t+95 ]

```

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```