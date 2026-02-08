model_name=TCN
model_id=NDA+TCN
dataset=ETTh1_dep
pred_lens=(96 192 336 720)
model_configs=(
    configs/models/ETTh1/TCN_0.yaml 
    configs/models/ETTh1/TCN_1.yaml 
    configs/models/ETTh1/TCN_2.yaml)

pivot=4
# python -u run_dep.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --data_name $dataset \
#     --model_configs ${model_configs[@]} \
#     --model_id $model_id \
#     --pivot $pivot \
#     --features S \
#     --target OT \
#     --model $model_name \
#     --seq_len 96 \
#     --label_len 0 \
#     --pred_len 336 \
#     --des 'Exp' \
#     --itr 1 



for i in 2 1 0; do
python -u tune_tcn.py \
    --is_training 1 \
    --task_name long_term_forecast \
    --model $model_name \
    --model_configs ${model_configs[@]} \
    --data_name $dataset \
    --features S \
    --target OT \
    --seq_len 96 \
    --pred_len 720 \
    --pivot $pivot \
    --train_component ${i} \
    --des 'Tuning_Exp' \
    --itr 1 
done