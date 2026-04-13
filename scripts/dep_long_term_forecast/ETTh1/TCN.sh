model_name=TCN
model_id=NDA+TCN
dataset=ETTh1_dep
pred_lens=(96)
model_configs=(
    configs/models/ETTh1/univariate/TCN_0.yaml 
    configs/models/ETTh1/univariate/TCN_1.yaml 
    configs/models/ETTh1/univariate/TCN_2.yaml)

pivot=7
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --model_id $model_id \
        --pivot $pivot \
        --features S \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp'
    
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --model_id $model_id \
        --pivot $pivot \
        --features S \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp'
done