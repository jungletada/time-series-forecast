export CUDA_VISIBLE_DEVICES=0
model_name=RNN
rnn_type=RNN
d_model=32
e_layers=3
dropout=0.1
pred_lens=(12 24 48)
features=S

# Optional override via command line: --rnn_type GRU (or --rnn_type=GRU)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rnn_type)
      shift
      rnn_type="${1:-$rnn_type}"
      ;;
    --rnn_type=*)
      rnn_type="${1#*=}"
      ;;
  esac
  shift
done

for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name PEMS03 \
    --model_id PEMS03_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
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
    --data_name PEMS04 \
    --model_id PEMS04_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
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
    --data_name PEMS07 \
    --model_id PEMS07_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
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
    --data_name PEMS08 \
    --model_id PEMS08_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
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