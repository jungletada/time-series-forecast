export CUDA_VISIBLE_DEVICES=0

model_name=TCN
dataset=ETTh1
seq_len=96
d_model=32
e_layers=3
pred_lens=(96 192 336 720)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_name $dataset \
#   --model_id ETTh1_$seq_len'_'192 \
#   --model $model_name \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_name $dataset \
#   --model_id ETTh1_$seq_len'_'336 \
#   --model $model_name \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 336 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_name $dataset \
#   --model_id ETTh1_$seq_len'_'720 \
#   --model $model_name \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 720 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window
