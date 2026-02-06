model_name=TimeFilter
model_id=NDA+TimeFilter
dataset=PEMS03_dep

d_ffs=(512 512 1024 512)
patch_lens=(48 48 96 48)
use_norm=(1 0 0 1)

model_configs=(
    configs/models/PEMS03/Timefilter_0.yaml 
    configs/models/PEMS03/Timefilter_1.yaml 
    configs/models/PEMS03/Timefilter_2.yaml)

pivot=4
python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id $model_id \
    --model $model_name \
    --model_configs ${model_configs[@]} \
    --pivot $pivot \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --print_freq 10 \
    --des 'Exp' \
    --itr 1

