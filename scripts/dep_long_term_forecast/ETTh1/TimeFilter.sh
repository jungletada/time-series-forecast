dataset=ETTh1_dep
model_name=TimeFilter
model_configs=(
    configs/models/ETTh1/Timefilter_0.yaml 
    configs/models/ETTh1/Timefilter_1.yaml 
    configs/models/ETTh1/Timefilter_2.yaml)
    
pred_lens=(96 192 336 720)
best_pivot=(3 4 4 4)

pivot=4
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
    --is_training 1 \
    --task_name long_term_forecast \
    --model $model_name \
    --model_configs ${model_configs[@]} \
    --data_name $dataset \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --pivot $pivot \
    --des 'Exp' \
    --itr 1
done

# pivot=3    # 选择的分解主元
# for i in 2 1 0; do
# python -u tune.py \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model $model_name \
#     --model_configs ${model_configs[@]} \
#     --data_name $dataset \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --pivot ${pivot} \
#     --train_component ${i} \
#     --des 'Tuning_Exp' \
#     --itr 1 
# done
