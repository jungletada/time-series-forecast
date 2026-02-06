dataset=ETTh1_dep
model_name=TimeFilter
model_configs=(
    configs/models/ETTh1/Timefilter_0.yaml 
    configs/models/ETTh1/Timefilter_1.yaml 
    configs/models/ETTh1/Timefilter_2.yaml)

# for pivot in 3; do
# python -u run_dep.py \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model $model_name \
#     --model_configs ${model_configs[@]} \
#     --data_name $dataset \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --pivot $pivot \
#     --des 'Exp' \
#     --itr 1
# done

pivot=3    # 选择的分解主元
for i in 0; do
python -u tune.py \
    --is_training 1 \
    --task_name long_term_forecast \
    --model $model_name \
    --model_configs ${model_configs[@]} \
    --data_name $dataset \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --pivot ${pivot} \
    --train_component ${i} \
    --des 'Tuning_Exp' \
    --itr 1 
done
