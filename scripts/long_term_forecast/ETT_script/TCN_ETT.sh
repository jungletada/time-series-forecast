export CUDA_VISIBLE_DEVICES=0

model_name=TCN
seq_len=96
d_model=16
e_layers=4
pred_lens=(96 192 336 720)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTh1 \
    --model_id ETTh1_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --enc_in 7 \
    --c_out 7 \
    --d_model $d_model \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128
done

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTh2 \
    --model_id ETTh2_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --enc_in 7 \
    --c_out 7 \
    --d_model $d_model \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128
done

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTm1 \
    --model_id ETTm1_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --enc_in 7 \
    --c_out 7 \
    --d_model $d_model \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128
done

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTm2 \
    --model_id ETTm2_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --enc_in 7 \
    --c_out 7 \
    --d_model $d_model \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128
done