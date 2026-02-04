model_name=TimeFilter_NDA
model_id=TimeFilter_NDA
pred_lens=(96 192 336 720)
dropout=(0.8 0.8 0.8 0.8)
patch_len=(2 2 2 2)
d_model=(128 128 128 128)
d_ff=(256 256 256 128)

for i in 96; do
python -u run.py \
  --seed 3407 \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name ETTh1 \
  --model_id $model_id \
  --model $model_name \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $i \
  --dropout 0.9 \
  --patch_len 16 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --pos 0 \
  --d_model 96 \
  --d_ff 96 \
  --moe_weight 0.1 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 10 \
  --des 'Exp' \
  --print_freq 100 \
  --itr 1
done

# for i in 720; do
# python -u run.py \
#   --seed 3407 \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_name ETTh1_dep \
#   --model_id $model_name \
#   --model $model_id \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 48 \
#   --pred_len $i \
#   --dropout 0.9 \
#   --patch_len 2 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --pos 0 \
#   --d_model 128 \
#   --d_ff 128 \
#   --moe_weight 0.05 \
#   --learning_rate 0.0001 \
#   --batch_size 32 \
#   --train_epochs 10 \
#   --patience 10 \
#   --des 'Exp' \
#   --itr 1
# done