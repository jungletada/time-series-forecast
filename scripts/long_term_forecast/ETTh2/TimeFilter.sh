export CUDA_VISIBLE_DEVICES=0
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \

model_name=TimeFilter
pred_lens=(96 192 336 720)
dropout=(0.8 0.6 0.7 0.3)
alpha=(0.1 0.8 0.4 0.9)
d_model=(128 128 256 256)
d_ff=(256 256 256 256)

for i in "${!pred_lens[@]}"; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name ETTh2 \
  --model_id ETTh2_${pred_lens[$i]} \
  --model $model_name \
  --features S \
  --target OT \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_len 4 \
  --alpha ${alpha[$i]} \
  --dropout ${dropout[$i]} \
  --top_p 0.0 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${pred_lens[$i]} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --learning_rate 0.0001 \
  --d_model ${d_model[$i]} \
  --d_ff ${d_ff[$i]} \
  --des 'Exp' \
  --itr 1
done