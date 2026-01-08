export CUDA_VISIBLE_DEVICES=0

model_name=TimeFilter
seq_len=96

pred_lens=(12 24 48)
for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name PEMS04 \
    --model_id PEMS04_96_${pred_lens[$i]} \
    --model $model_name \
    --features MS \
    --target 97 \
    --seq_len 96 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --dropout 0.1 \
    --top_p 0.0 \
    --learning_rate 0.0005 \
    --batch_size 16 \
    --train_epochs 20 \
    --itr 1 \
    --use_norm 0
done

