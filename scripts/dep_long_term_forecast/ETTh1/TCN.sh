model_name=TCN
model_id=NDA+TCN
dataset=ETTh1_dep
seq_len=96
d_model=32
e_layers=4
mnn=mlp
pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model_id $model_id \
        --features S \
        --model $model_name \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 10 \
        --patience 5 \
        --batch_size 16
    
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --model_id $model_id \
        --features S \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp' \
        --batch_size 32 \
        --itr 1
done
