d_model=32
e_layers=4
model=TCN
mnn=mlp
seq_len=96
dataset=Exchange_dep

pred_lens=(96 192 336 720)
for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model $model \
        --features S \
        --data_name $dataset \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.005 \
        --train_epochs 10 \
        --patience 3 \
        --batch_size 32

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --mnn $mnn \
        --features S \
        --data_name $dataset \
        --model $model \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --batch_size 32 \
        --des 'Exp' \
        --itr 1
done