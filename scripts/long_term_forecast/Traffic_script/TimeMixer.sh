export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer
seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=8
pred_lens=(96 192 336 720)
# --enc_in 862 \
# --dec_in 862 \
# --c_out 862 \

for i in "${!pred_lens[@]}"; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name Traffic \
  --model_id traffic_$seq_len'_'${pred_lens[$i]} \
  --model $model_name \
  --features S \
  --target OT \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len ${pred_lens[$i]} \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_method avg \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_window $down_sampling_window
done
