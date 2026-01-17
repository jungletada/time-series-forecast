model_name=TimeFilter
pred_lens=(24 36 48 60)
d_model=(128 128 128 128)
d_ff=(512 512 512 512)
dropout=(0.3 0.3 0.3 0.3)
patch_len=(36 36 36 36)


for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --data_name Illness \
    --model $model_name \
    --model_id $model_name \
    --features S \
    --seq_len 36 \
    --label_len 18 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch_len ${patch_len[$i]} \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 10 \
    --des 'Exp' \
    --d_model ${d_model[$i]}\
    --d_ff ${d_ff[$i]}\
    --dropout ${dropout[$i]} \
    --itr 1
done
