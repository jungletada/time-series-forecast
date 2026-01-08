export CUDA_VISIBLE_DEVICES=7

model_name=TimeFilter
seq_len=96

pred_lens=(12 24 48)
for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name PEMS08 \
    --model_id PEMS08_96_${pred_lens[$i]} \
    --model $model_name \
    --features MS \
    --target 9 \
    --seq_len 96 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --dropout 0.1 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 20 \
    --itr 1 \
    --use_norm 1
done
