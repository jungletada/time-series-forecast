export CUDA_VISIBLE_DEVICES=0

e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTm2 \
    --model_id ETTm2_96'_'${pred_lens[$i]} \
    --model TimeMixer \
    --features MS \
    --target OT \
    --enc_in 7 \
    --c_out 7 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 16 \
    --learning_rate $learning_rate \
    --down_sampling_method avg \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_window $down_sampling_window
done
