dataset=ETTh1_dep
model_name=TimeFilter
model_configs=(
    configs/models/ETTh1/Timefilter_0.yaml 
    configs/models/ETTh1/Timefilter_1.yaml 
    configs/models/ETTh1/Timefilter_2.yaml)
    
pred_lens=(96 192 336 720)

python -u run_dep.py \
    --is_training 1 \
    --task_name long_term_forecast \
    --model $model_name \
    --model_configs ${model_configs[@]} \
    --data_name $dataset \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --pivot 3 \
    --des 'Exp' \
    --itr 1
