model_name=TimeMixer
dataset=Exchange_dep
d_ff=(16 16 64 16)
batch_size=128
train_epochs=10
patience=5
learning_rate=0.01
d_model=(32 32 32 32)
e_layers=3
down_sampling_layers=3
down_sampling_window=2
pred_lens=(96 192 336 720)


for i in "${!pred_lens[@]}"; do
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model $model_name \
    --features S \
    --seq_len 96 \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --des 'Exp' \
    --itr 1 \
    --d_model ${d_model[$i]} \
    --d_ff ${d_ff[$i]} \
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
    --data_name $dataset \
    --model $model_name \
    --features S \
    --seq_len 96 \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --des 'Exp' \
    --itr 1 \
    --d_model ${d_model[$i]} \
    --d_ff ${d_ff[$i]} \
    --batch_size $batch_size \
    --down_sampling_method avg \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_window $down_sampling_window
done
