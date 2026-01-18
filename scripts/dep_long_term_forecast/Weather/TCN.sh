export CUDA_VISIBLE_DEVICES=0
d_model=32
e_layers=4
model_id=NDA+TCN
pred_lens=(96 192 336 720)

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --use_mnn 0 \
        --data_name Weather_dep \
        --model_id $model_id \
        --model TCN \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --enc_in 1 \
        --c_out 1 \
        --target OT \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.01 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 128
done


for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name Weather_dep \
        --model_id $model_id \
        --model TCN \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --enc_in 1 \
        --c_out 1 \
        --target OT \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1 
done