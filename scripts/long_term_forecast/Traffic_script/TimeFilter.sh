model_name=TimeFilter

d_model=(512 512 512 512)
d_ff=(2048 2048 2048 2048)
dropout=(0.3 0.3 0.3 0.3)
patch_len=(96 96 96 96)

pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Traffic \
    --model_id $model_name \
    --model $model_name \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch_len ${patch_len[$i]} \
    --des 'Exp' \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 30 \
    --d_model ${d_model[$i]}\
    --d_ff ${d_ff[$i]}\
    --dropout ${dropout[$i]} \
    --itr 1
done
