export CUDA_VISIBLE_DEVICES=0
model_name=TCN
dataset=ETTh2_dep
seq_len=96
d_model=16
e_layers=4
pred_lens=(96 192 336 720)

# # Loop over datasets and prediction lengths
# for i in "${!pred_lens[@]}"; do
#     python -u run_dep.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --use_mnn 0 \
#         --data_name $dataset \
#         --model_id ETTh2_$seq_len'_'${pred_lens[$i]} \
#         --model $model_name \
#         --seq_len $seq_len \
#         --label_len 0 \
#         --pred_len ${pred_lens[$i]} \
#         --enc_in 1 \
#         --c_out 1 \
#         --target OT \
#         --features S \
#         --d_model $d_model \
#         --e_layers $e_layers \
#         --des 'Exp' \
#         --itr 1 \
#         --learning_rate 0.002 \
#         --train_epochs 10 \
#         --patience 10 \
#         --batch_size 128
# done

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model_id ETTh2_$seq_len'_'${pred_lens[$i]} \
        --model $model_name \
        --seq_len $seq_len \
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
        --learning_rate 0.002 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 128
done