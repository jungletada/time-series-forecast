d_model=32
e_layers=4
pred_lens=(96 192 336 720)
dataset=ETTm1_dep

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model TCN \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.008 0.004 0.002 \
        --train_epochs 10 \
        --patience 5 \
        --batch_size 32

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model TCN \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --batch_size 32 \
        --itr 1
done
