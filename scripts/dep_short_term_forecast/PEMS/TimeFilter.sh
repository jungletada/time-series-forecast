model_name=TimeFilter
model_id=NDA+TimeFilter
d_model=512
e_layers=2
dropout=0.1
features=S

d_ffs=(512 512 1024 512)
patch_lens=(48 48 96 48)
use_norm=(1 0 0 1)

# pred_lens=(12)
# dataset=PEMS03_dep
# model_configs=(configs/models/PEMS03/Timefilter_0.yaml configs/models/PEMS03/Timefilter_1.yaml configs/models/PEMS03/Timefilter_2.yaml)
# for i in "${!pred_lens[@]}"; do
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
#         --model_id $model_id \
#         --model $model_name \
#         --model_configs ${model_configs[@]} \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --des 'Exp' \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --patience 10 \
#         --itr 1

#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name $dataset \
#         --model_id $model_id \
#         --model $model_name \
#         --model_configs ${model_configs[@]} \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

# pred_lens=(12)
# dataset=PEMS04_dep
# model_configs=(configs/models/PEMS04/Timefilter_0.yaml configs/models/PEMS04/Timefilter_1.yaml configs/models/PEMS04/Timefilter_2.yaml)
# for i in "${!pred_lens[@]}"; do
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
#         --model_id $model_id \
#         --model $model_name \
#         --model_configs ${model_configs[@]} \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --des 'Exp' \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --patience 10 \
#         --itr 1

#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name $dataset \
#         --model_id $model_id \
#         --model $model_name \
#         --model_configs ${model_configs[@]} \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

# pred_lens=(12)
# dataset=PEMS07_dep
# model_configs=(configs/models/PEMS07/Timefilter_0.yaml configs/models/PEMS07/Timefilter_1.yaml configs/models/PEMS07/Timefilter_2.yaml)
# for i in "${!pred_lens[@]}"; do
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
#         --model_id $model_id \
#         --model $model_name \
#         --model_configs ${model_configs[@]} \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --des 'Exp' \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --patience 10 \
#         --itr 1

#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name $dataset \
#         --model_id $model_id \
#         --model $model_name \
#         --model_configs ${model_configs[@]} \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

pred_lens=(12)
dataset=PEMS08_dep
model_configs=(configs/models/PEMS08/Timefilter_0.yaml configs/models/PEMS08/Timefilter_1.yaml configs/models/PEMS08/Timefilter_2.yaml)
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model_id $model_id \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --features $features \
        --seq_len 96 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp' \
        --batch_size 16 \
        --train_epochs 20 \
        --patience 10 \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model_id $model_id \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --features $features \
        --seq_len 96 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1
done