model_name=RNN
rnn_type=RNN
d_model=64
e_layers=3
dropout=0.1
pred_lens=(96 192 336 720)
features=S

for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTh1 \
    --model_id ETTh1_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
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

for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTh2 \
    --model_id ETTh2_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
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

for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTm1 \
    --model_id ETTm1_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
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

for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTm2 \
    --model_id ETTm2_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
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