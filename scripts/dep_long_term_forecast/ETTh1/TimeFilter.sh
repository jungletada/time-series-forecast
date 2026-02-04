
model_name=TimeFilter
model_id=NDA+TimeFilter
dataset=ETTh1_dep
model_configs=(configs/models/ETTh1/Timefilter_0.yaml configs/models/ETTh1/Timefilter_1.yaml configs/models/ETTh1/Timefilter_2.yaml)

pred_lens=(720)
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --is_training 1 \
        --task_name long_term_forecast \
        --model_configs ${model_configs[@]} \
        --data_name $dataset \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp' \
        --itr 1
    
    # python -u run_dep.py \
    #     --task_name long_term_forecast \
    #     --is_training 0 \
    #     --use_mnn 1 \
    #     --data_name $dataset \
    #     --model_id $model_id \
    #     --model $model_name \
    #     --model_configs ${model_configs[@]} \
    #     --features S \
    #     --seq_len 96 \
    #     --label_len 48 \
    #     --pred_len ${pred_lens[$i]} \
    #     --batch_size 16 \
    #     --des 'Exp' \
    #     --itr 1
done