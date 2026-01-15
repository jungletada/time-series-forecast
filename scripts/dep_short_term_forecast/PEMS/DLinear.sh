model=DLinear
pred_lens=(12 24 48)
datasets=(PEMS03_dep PEMS04_dep PEMS07_dep PEMS08_dep)

for dataset in "${datasets[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
        python -u run_dep.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --data_name $dataset \
            --model $model \
            --features S \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pred_lens[$i]} \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --des 'Exp' \
            --itr 1

            python -u run_dep.py \
            --task_name long_term_forecast \
            --is_training 0 \
            --use_mnn 1 \
            --mnn mlp \
            --data_name $dataset \
            --model $model \
            --features S \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pred_lens[$i]} \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --des 'Exp' \
            --itr 1
done 