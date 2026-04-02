export CUDA_VISIBLE_DEVICES=0

pred_lens=(96 192 336 720)
for i in "${!pred_lens[@]}"; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name ETTm1 \
  --model_id ETTm1_96_${pred_lens[$i]} \
  --model DLinear \
  --features S \
  --target OT \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_lens[$i]} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --des 'Exp' \
  --itr 1
done