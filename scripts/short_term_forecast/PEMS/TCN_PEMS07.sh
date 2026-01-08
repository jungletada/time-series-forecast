export CUDA_VISIBLE_DEVICES=0

model_name=TCN
seq_len=96
pred_lens=(12 24 48)
d_model=36
e_layers=4
# --enc_in 358 \
# --dec_in 358 \
# --c_out 358 \

for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS07 \
    --model_id PEMS07_96_${pred_lens[$i]} \
    --model $model_name \
    --features S \
    --seq_len 96 \
    --pred_len ${pred_lens[$i]} \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model $d_model \
    --e_layers $e_layers \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 20 \
    --itr 1
done
