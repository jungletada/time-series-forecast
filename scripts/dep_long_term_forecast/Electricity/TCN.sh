seq_len=96
d_model=32
e_layers=4
model_name=TCN
model_id=NDA+TCN
dataset=Electricity_dep
pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --model_id $model_id \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.008 0.004 0.002 \
        --train_epochs 10 \
        --patience 5 \
        --batch_size 16

    python -u run_dep.py \
        --task_name long_term_forecast \
        --model_id $model_id \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --batch_size 32 \
        --des 'Exp' \
        --itr 1 
done
