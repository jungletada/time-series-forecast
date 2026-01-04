export CUDA_VISIBLE_DEVICES=0

d_model=32
e_layers=4
pred_lens=(24 36 48 60)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Illness \
    --model_id Ill_36_${pred_lens[$i]} \
    --model TCN \
    --features S \
    --target OT \
    --seq_len 36 \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --enc_in 1 \
    --c_out 1 \
    --d_model $d_model \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.005 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16
done