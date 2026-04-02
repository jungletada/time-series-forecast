export CUDA_VISIBLE_DEVICES=2

model_name=TimesNet

seq_len=96
pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTh1 \
    --model_id ETTh1_$seq_len'_'${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --target OT \
    --enc_in 1 \
    --c_out 1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 1 \
    --top_k 5 
done