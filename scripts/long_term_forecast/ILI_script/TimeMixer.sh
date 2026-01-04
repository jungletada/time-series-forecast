export CUDA_VISIBLE_DEVICES=0
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=32
train_epochs=20
patience=10
pred_lens=(24 36 48 60)

# For MS, M:
#   --enc_in 7 \ 
#   --dec_in 7 \  
#   --c_out 7 \   

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Illness \
    --model_id Ill_36_${pred_lens[$i]} \
    --model TimeMixer \
    --features S \
    --target OT \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --seq_len 36 \
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
