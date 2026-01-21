
model_name=TimeFilter
model_id=NDA+TimeFilter
dataset=ETTh2_dep
dropout=(0.8 0.8 0.8 0.8)
patch_len=(2 2 2 2)

d_model=(128 128 128 128)
d_ff=(256 256 256 128)
learning_rate=0.0001
model_configs=(configs/models/ETTh2/Timefilter_0.yaml configs/models/ETTh2/Timefilter_1.yaml configs/models/ETTh2/Timefilter_2.yaml)

pred_lens=(720)
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model_id $model_id \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --features S \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pred_lens[$i]} \
        --batch_size 128 \
        --train_epochs 10 \
        --print_freq 10 \
        --des 'Exp' \
        --itr 1
    
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model_id $model_id \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --features S \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pred_lens[$i]} \
        --batch_size 16 \
        --des 'Exp' \
        --itr 1
done