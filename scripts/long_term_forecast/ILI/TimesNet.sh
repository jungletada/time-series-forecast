export CUDA_VISIBLE_DEVICES=0

pred_lens=(24 36 48 60)

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Illness \
    --model_id Ill_36_${pred_lens[$i]} \
    --model TimesNet \
    --features S \
    --target OT \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --seq_len 36 \
    --label_len 18 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 768 \
    --d_ff 768 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1
done

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_36_36 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 36 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 768 \
#   --d_ff 768 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_36_48 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 768 \
#   --d_ff 768 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_36_60 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 768 \
#   --d_ff 768 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1