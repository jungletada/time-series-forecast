export CUDA_VISIBLE_DEVICES=0
model_name=TimeXer
pred_lens=(96 192 336 720)
  # --enc_in 862 \
  # --dec_in 862 \
  # --c_out 862 \
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Traffic \
    --model_id traffic_96_${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --target OT \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 3 \
    --factor 3 \
    --d_model 512 \
    --d_ff 512 \
    --des 'Exp' \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1
done

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 3 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --d_model 512 \
#   --d_ff 512 \
#   --des 'Exp' \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --d_model 512 \
#   --d_ff 512 \
#   --des 'Exp' \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --d_model 512 \
#   --d_ff 512 \
#   --des 'Exp' \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1
