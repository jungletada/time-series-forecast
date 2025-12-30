export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer
seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=16
pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTm2 \
    --model_id ETTm2_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --target OT \
    --enc_in 1 \
    --c_out 1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --enc_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 128 \
    --learning_rate $learning_rate \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window
done

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path  ./dataset/ETT-small/\
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'96 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'192 \
#   --model $model_name \
#   --data ETTm2 \
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
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'336 \
#   --model $model_name \
#   --data ETTm2 \
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
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'720 \
#   --model $model_name \
#   --data ETTm2 \
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
#   --batch_size 128 \
#   --learning_rate $learning_rate \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window
