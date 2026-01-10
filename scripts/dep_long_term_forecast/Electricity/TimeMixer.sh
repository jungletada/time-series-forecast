#export CUDA_VISIBLE_DEVICES=0
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=32
train_epochs=20
patience=10
pred_lens=(96 192 336 720)
features=S

for i in "${!pred_lens[@]}"; do
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Electricity_dep \
    --model_id ECL_dep_96_${pred_lens[$i]} \
    --model TimeMixer \
    --features $features \
    --seq_len 96 \
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
    --train_epochs $train_epochs \
    --patience $patience \
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
    --data_name Electricity_dep \
    --model_id ECL_dep_96_${pred_lens[$i]} \
    --model TimeMixer \
    --features $features \
    --seq_len 96 \
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
    --down_sampling_method avg \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_window $down_sampling_window
done