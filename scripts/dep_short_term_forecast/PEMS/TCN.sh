model_name=TCN
seq_len=96
d_model=36
e_layers=4
features=S

pred_lens=(48)
dataset=PEMS03_dep
model_configs=(configs/models/PEMS03/TCN_0.yaml configs/models/PEMS03/TCN_1.yaml configs/models/PEMS03/TCN_2.yaml)
for pred_len in "${pred_lens[@]}"; do
    # python run_dep.py \
    #     --task_name long_term_forecast \
    #     --is_training 1 \
    #     --data_name $dataset \
    #     --model_configs ${model_configs[@]} \
    #     --model $model_name \
    #     --model_id $model_name \
    #     --features $features \
    #     --seq_len 96 \
    #     --pred_len $pred_len \
    #     --des 'Exp' \
    #     --batch_size 16 \
    #     --train_epochs 20 \
    #     --patience 10 \
    #     --itr 1

    python run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model_configs ${model_configs[@]} \
        --model $model_name \
        --model_id $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len $pred_len \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1
done


# pred_lens=(12)
# dataset=PEMS04_dep
# model_configs=(configs/models/PEMS04/TCN_0.yaml configs/models/PEMS04/TCN_1.yaml configs/models/PEMS04/TCN_2.yaml)
# for pred_len in "${pred_lens[@]}"; do
#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
#         --model_configs ${model_configs[@]} \
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --train_epochs 10 \
#         --patience 5 \
#         --itr 1

#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name $dataset \
#         --model_configs ${model_configs[@]} \
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

# pred_lens=(12)
# dataset=PEMS07_dep
# model_configs=(configs/models/PEMS07/TCN_0.yaml configs/models/PEMS07/TCN_1.yaml configs/models/PEMS07/TCN_2.yaml)
# for pred_len in "${pred_lens[@]}"; do
#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
#         --model_configs ${model_configs[@]} \
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --train_epochs 10 \
#         --patience 5 \
#         --itr 1

#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name $dataset \
#         --model_configs ${model_configs[@]} \
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

# pred_lens=(12)
# dataset=PEMS08_dep
# model_configs=(configs/models/PEMS08/TCN_0.yaml configs/models/PEMS08/TCN_1.yaml configs/models/PEMS08/TCN_2.yaml)
# for pred_len in "${pred_lens[@]}"; do
    # python run_dep.py \
    #     --task_name long_term_forecast \
    #     --is_training 1 \
    #     --data_name $dataset \
    #     --model_configs ${model_configs[@]} \
    #     --model $model_name \
    #     --model_id $model_name \
    #     --features $features \
    #     --seq_len 96 \
    #     --pred_len $pred_len \
    #     --des 'Exp' \
    #     --batch_size 16 \
    #     --train_epochs 10 \
    #     --patience 5 \
    #     --itr 1

#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name $dataset \
#         --model_configs ${model_configs[@]} \
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done