export CUDA_VISIBLE_DEVICES=0
model_name=Transformer
pred_lens=(96 192 336 720)
features=S
for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Electricity \
    --model_id ECL_96_$pred_len \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --patience 5 \
    --itr 1
done