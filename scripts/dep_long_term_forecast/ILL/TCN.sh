d_model=32
e_layers=4
features=S
pred_lens=(24 36 48 60)

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --use_mnn 0 \
        --data_name Illness_dep \
        --model TCN \
        --features $features \
        --seq_len 36 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.01 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 32

        python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name Illness_dep \
        --model TCN \
        --features $features \
        --seq_len 36 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --batch_size 32 \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1  
done
