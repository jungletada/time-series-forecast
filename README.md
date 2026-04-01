# Time Series Library (TSLib)
-----
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.
Modified by the team of Waseda University, Tokyo, Japan
-----
## Getting Started

### Prepare Data
You can obtain the well-preprocessed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library). Then place the downloaded data in the folder `dataset`.

### Installation
1. Clone this repository.
   ```bash
   git clone https://github.com/jungletada/time-series-forecast.git
   cd time-series-forecast
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

4. Install Dependencies for Mamba Model (Required for time-series-forecast/models/Mamba.py)
   > ⚠️ **CUDA Compatibility Notice**
   > The prebuilt Mamba wheel is **CUDA-version specific**.
   > Please make sure to install the wheel that matches your local CUDA version
   > (e.g., `cu11` or `cu12`). Installing a mismatched version may result in
   > runtime errors or import failures.

   Example for **CUDA 12**:

   ```bash
   pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   ```

5. Install Dependencies for Moirai Model (Required for time-series-forecast/models/Moirai.py)
   ```bash
   pip install uni2ts --no-deps
   ```

### Docker Deployment
```bash
# Build and start the Docker container in detached mode
docker compose -f 'time-series-forecast/docker-compose.yml' up -d --build

# Download / place the dataset into a newly created folder ./dataset at the repository root
mkdir -p dataset  # create the dataset directory

# Copy the local dataset into the container at /workspace/dataset
docker cp ./dataset tslib:/workspace/dataset

# Enter the running container to continue training / evaluation
docker exec -it tslib bash

# Switch to the workspace directory inside the container
cd /workspace
```

### Train and Evaluate
We provide the experiment scripts for all benchmarks under the folder `scripts/`. You can reproduce the experiment results as the following examples:

```bash
# long-term forecast
bash scripts/long_term_forecast/ETTh1_script/TimesNet.sh
# short-term forecast
bash scripts/short_term_forecast/TimesNet_M4.sh
```

Run TimeFilter
```bash
bash scripts/long_term_forecast/run_TimeFilter.sh
```

### Develop Your Own Model
- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `scripts`.

### Inspect the project structure:

```
time-series-forecast/
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

## 新增内容：NDA
- NDA + Timefilter
```bash
bash scripts/dep_long_term_forecast/ETTh1/TimeFilter.sh
```
会自动读取`configs/models/ETTh1`里面的 `Timefilter_0.yaml`，`Timefilter_1.yaml`，`Timefilter_2.yaml`

- NDA + WPMixer
```bash
bash scripts/dep_long_term_forecast/ETTh1/WPMixer.sh
```
注意文件夹`configs/models`里面的下不一定有`WPMixer_0.yaml`, `WPMixer_1.yaml`, `WPMixer_2.yaml`，需要复制`WPMixer.yaml`再自行创建。

- NDA + FreEformer
```bash
bash scripts/dep_long_term_forecast/ETTh1/FreEformer.sh
```
注意文件夹`configs/models`里面的下不一定有`FreEformer_0.yaml`, `FreEformer_1.yaml`, `FreEformer_2.yaml`，需要复制`FreEformer.yaml`再自行创建。


- NDA + PatchMLP
```bash
bash scripts/dep_long_term_forecast/ETTh1/PatchMLP.sh
```
注意文件夹`configs/models`里面的下不一定有`PatchMLP_0.yaml`, `PatchMLP_1.yaml`, `PatchMLP_2.yaml`，需要复制`PatchMLP.yaml`再自行创建。


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