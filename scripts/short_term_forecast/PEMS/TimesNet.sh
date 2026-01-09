pred_lens=(12 24 48)
features=S

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS03 \
    --model_id PEMS03_96_${pred_lens[$i]} \
    --model TimesNet \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 256 \
    --d_ff 512 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1
done

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS04 \
    --model_id PEMS04_96_${pred_lens[$i]} \
    --model TimesNet \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 256 \
    --d_ff 512 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1
done

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS07 \
    --model_id PEMS07_96_${pred_lens[$i]} \
    --model TimesNet \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 256 \
    --d_ff 512 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1
done

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS08 \
    --model_id PEMS08_96_${pred_lens[$i]} \
    --model TimesNet \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 256 \
    --d_ff 512 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1
done