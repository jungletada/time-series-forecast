export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer
dataset=ETTm2_dep
seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
d_model=16
d_ff=32
pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id ETTm2_dep_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --target OT \
    --enc_in 1 \
    --c_out 1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128 \
    --down_sampling_method avg \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_window $down_sampling_window
done


for i in "${!pred_lens[@]}"; do
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --mnn mlp \
    --data_name $dataset \
    --model_id ETTm2_dep_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --target OT \
    --enc_in 1 \
    --c_out 1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 128 \
    --down_sampling_method avg \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_window $down_sampling_window
done