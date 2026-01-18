model_name=TCN
seq_len=96
d_model=36
e_layers=4
features=S
pred_lens=(12 24 48)


dataset=PEMS03_dep
for pred_len in "${pred_lens[@]}"; do
    python run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --model_id $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len $pred_len \
        --des 'Exp' \
        --batch_size 32 \
        --train_epochs 10 \
        --patience 5 \
        --itr 1

    python run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --model_id $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len $pred_len \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1
done


# dataset=PEMS03_dep
# for pred_len in "${pred_lens[@]}"; do
#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
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
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

# dataset=PEMS04_dep
# for pred_len in "${pred_lens[@]}"; do
#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
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
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

# dataset=PEMS07_dep
# for pred_len in "${pred_lens[@]}"; do
#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
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
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done

# dataset=PEMS08_dep
# for pred_len in "${pred_lens[@]}"; do
#     python run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --data_name $dataset \
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
#         --model $model_name \
#         --model_id $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len $pred_len \
#         --des 'Exp' \
#         --batch_size 16 \
#         --itr 1
# done