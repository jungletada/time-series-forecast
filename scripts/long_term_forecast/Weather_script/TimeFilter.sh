export CUDA_VISIBLE_DEVICES=0
model_name=TimeFilter
pred_lens=(96 192 336 720)
d_model=(128 128 128 128)
d_ff=(256 256 256 256)
dropout=(0.3 0.3 0.3 0.3)
patch_len=(48 48 48 48)

# --enc_in 21 \
# --dec_in 21 \
# --c_out 21 \

for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Weather \
    --model_id Weather_96_${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --target OT \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch_len ${patch_len[$i]} \
    --des 'Exp' \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --train_epochs 10 \
    --d_model ${d_model[$i]}\
    --d_ff ${d_ff[$i]}\
    --dropout ${dropout[$i]} \
    --itr 1
done
