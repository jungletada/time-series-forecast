

model_name=Transformer
pred_lens=(24 36 48 60)
features=S

for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name Illness \
  --model $model_name \
  --features $features \
  --seq_len 36 \
  --label_len 18 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 5 \
  --des 'Exp' \
  --itr 1 
done