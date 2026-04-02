
model_name=PatchTST
model_configs=(configs/models/Illness/PatchTST_0.yaml configs/models/Illness/PatchTST_1.yaml configs/models/Illness/PatchTST_2.yaml)

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_36_24 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 24 \
#   --e_layers 4 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_heads 4 \
#   --d_model 1024\
#   --itr 1

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
#   --e_layers 4 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_heads 4 \
#   --d_model 2048\
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
#   --e_layers 4 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_heads 4 \
#   --d_model 2048\
#   --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name Illness \
  --model_id $model_name \
  --model $model_name \
  --features S \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --des 'Exp' \
  --n_heads 16 \
  --d_model 2048 \
  --train_epochs 10 \
  --learning_rate 0.0001 \
  --batch_size 8 \
  --itr 1