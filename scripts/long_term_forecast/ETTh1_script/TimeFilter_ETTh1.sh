model_name=TimeFilter
model_id=TimeFilter
seq_len=96
pred_lens=(96 192 336 720)
dropout=(0.8 0.8 0.8 0.8)
patch_len=(2 2 2 2)
d_model=(128 128 128 128)
d_ff=(256 256 256 128)

for i in "${!pred_lens[@]}"; do
python -u run.py \
  --seed 3407 \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name ETTh1 \
  --model_id $model_name \
  --model $model_id \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len ${pred_lens[$i]} \
  --dropout ${dropout[$i]} \
  --patch_len ${patch_len[$i]} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --pos 0 \
  --d_model ${d_model[$i]} \
  --d_ff ${d_ff[$i]} \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 10 \
  --des 'Exp' \
  --itr 1
done