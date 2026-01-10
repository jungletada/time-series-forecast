model_name=RNN
rnn_type=RNN
d_model=64
e_layers=3
dropout=0.1
pred_lens=(24 36 48 60)
features=S

for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Illness \
    --model_id Ill_36_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 36 \
    --label_len 0 \
    --pred_len $pred_len \
    --d_model $d_model \
    --rnn_type $rnn_type \
    --e_layers $e_layers \
    --dropout $dropout \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --patience 5 \
    --itr 1
done