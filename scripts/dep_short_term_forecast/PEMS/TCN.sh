model_name=TCN
seq_len=96
d_model=36
e_layers=4
features=S
pred_lens=(12 24 48)
datasets=(PEMS03_dep PEMS04_dep PEMS07_dep PEMS08_dep)

for dataset in "${datasets[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --seed 5566 \
        --is_training 1 \
        --data_name $dataset \
        --model_id $dataset_96_$pred_len \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len $pred_len \
        --patch_len 48 \
        --des 'Exp' \
        --d_model $d_model \
        --e_layers $e_layers \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 20 \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --seed 5566 \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model_id $dataset_96_$pred_len \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len $pred_len \
        --patch_len 48 \
        --des 'Exp' \
        --d_model $d_model \
        --e_layers $e_layers \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 20 \
        --itr 1
    done
done

# for i in "${!pred_lens[@]}"; do
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --seed 5566 \
#         --is_training 1 \
#         --data_name PEMS04_dep \
#         --model_id PEMS04_96_${pred_lens[$i]} \
#         --model $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --patch_len 48 \
#         --des 'Exp' \
#         --d_model $d_model \
#         --e_layers $e_layers \
#         --learning_rate 0.001 \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --itr 1

#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --seed 5566 \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name PEMS04_dep \
#         --model_id PEMS04_96_${pred_lens[$i]} \
#         --model $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --patch_len 48 \
#         --des 'Exp' \
#         --d_model $d_model \
#         --e_layers $e_layers \
#         --learning_rate 0.001 \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --itr 1
# done

# for i in "${!pred_lens[@]}"; do
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --seed 5566 \
#         --is_training 1 \
#         --data_name PEMS07_dep \
#         --model_id PEMS07_96_${pred_lens[$i]} \
#         --model $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --patch_len 48 \
#         --des 'Exp' \
#         --d_model $d_model \
#         --e_layers $e_layers \
#         --learning_rate 0.001 \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --itr 1

#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --seed 5566 \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name PEMS07_dep \
#         --model_id PEMS07_96_${pred_lens[$i]} \
#         --model $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --patch_len 48 \
#         --des 'Exp' \
#         --d_model $d_model \
#         --e_layers $e_layers \
#         --batch_size 16 \
#         --itr 1
# done

# for i in "${!pred_lens[@]}"; do
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --seed 5566 \
#         --is_training 1 \
#         --data_name PEMS08_dep \
#         --model_id PEMS08_96_${pred_lens[$i]} \
#         --model $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --patch_len 48 \
#         --des 'Exp' \
#         --d_model $d_model \
#         --e_layers $e_layers \
#         --learning_rate 0.001 \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --itr 1
    
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --seed 5566 \
#         --is_training 0 \
#         --use_mnn 1 \
#         --data_name PEMS08_dep \
#         --model_id PEMS08_96_${pred_lens[$i]} \
#         --model $model_name \
#         --features $features \
#         --seq_len 96 \
#         --pred_len ${pred_lens[$i]} \
#         --patch_len 48 \
#         --des 'Exp' \
#         --d_model $d_model \
#         --e_layers $e_layers \
#         --learning_rate 0.001 \
#         --batch_size 16 \
#         --train_epochs 20 \
#         --itr 1
# done
