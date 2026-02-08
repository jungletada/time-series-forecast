model_name=TimeFilter
model_id=NDA+TimeFilter
dataset=PEMS08_dep

model_configs=(
    configs/models/PEMS08/Timefilter_0.yaml 
    configs/models/PEMS08/Timefilter_1.yaml 
    configs/models/PEMS08/Timefilter_2.yaml)

pivot=4
pred_lens=(12 24 48)
for i in "${!pred_lens[@]}"; do
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
    --pred_len ${pred_lens[$i]} \
    --print_freq 10 \
    --des 'Exp' \
    --itr 1
done

