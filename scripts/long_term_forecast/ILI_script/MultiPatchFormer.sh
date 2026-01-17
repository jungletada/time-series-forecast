model_name=MultiPatchFormer

pred_lens=(24 36 48 60)
for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --data_name Illness \
    --model_id $model_name \
    --model $model_name \
    --features S \
    --seq_len 36 \
    --label_len 18 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 1 \
    --d_model 256 \
    --d_ff 512 \
    --des 'Exp' \
    --n_heads 8 \
    --batch_size 32 \
    --itr 1
done