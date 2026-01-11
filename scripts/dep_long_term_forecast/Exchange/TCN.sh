export CUDA_VISIBLE_DEVICES=0
d_model=32
e_layers=4
dataset=Exchange_dep
pred_lens=(96 192 336 720)
mnn=mlp
seq_len=96

for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model_id $dataset'_'$seq_len'_'$pred_len \
        --model TCN \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --features S \
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
        --mnn $mnn \
        --data_name $dataset \
        --model_id $dataset'_'$seq_len'_'$pred_len \
        --model TCN \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1
done