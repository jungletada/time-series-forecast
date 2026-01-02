export CUDA_VISIBLE_DEVICES=0

d_model=16
e_layers=4
pred_lens=(96 192 336 720)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Electricity \
    --model_id ECL_96'_'${pred_lens[$i]} \
    --model TCN \
    --features S \
    --target OT \
    --seq_len 96 \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --enc_in 1 \
    --c_out 1 \
    --d_model $d_model \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128
done