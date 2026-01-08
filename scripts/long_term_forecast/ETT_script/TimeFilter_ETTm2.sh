export CUDA_VISIBLE_DEVICES=0
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \

model_name=TimeFilter
pred_lens=(96 192 336 720)
pred_lens=(96 192 336 720)
dropout=(0.6 0.6 0.6 0.8)
top_p=(0.0 0.0 0.0 0.0)
patch_len=(16 16 16 16)
d_model=(128 128 128 128)
d_ff=(128 128 128 128)

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name ETTm1 \
    --model_id ETTm1_${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --target OT \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --dropout ${dropout[$i]} \
    --top_p ${top_p[$i]} \
    --patch_len ${patch_len[$i]} \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model ${d_model[$i]} \
    --d_ff ${d_ff[$i]} \
    --des 'Exp' \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --train_epochs 10 \
    --itr 1
done