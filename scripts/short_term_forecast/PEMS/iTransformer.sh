model_name=iTransformer
pred_lens=(12 24 48)
features=S

for i in "${!pred_lens[@]}"; do
    python run.py \
        --task_name long_term_forecast \
        --seed 5566 \
        --is_training 1 \
        --data_name PEMS03 \
        --model_id PEMS03_96_${pred_lens[$i]} \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len ${pred_lens[$i]} \
        --e_layers 4 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --learning_rate 0.001 \
        --itr 1
done

model_name=iTransformer
pred_lens=(12 24 48)

for i in "${!pred_lens[@]}"; do
    python run.py \
        --task_name long_term_forecast \
        --seed 5566 \
        --is_training 1 \
        --data_name PEMS04 \
        --model_id PEMS04_96_${pred_lens[$i]} \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len ${pred_lens[$i]} \
        --e_layers 4 \
        --des 'Exp' \
        --d_model 1024 \
        --d_ff 1024 \
        --learning_rate 0.005 \
        --itr 1 \
        --use_norm 0
done

e_layers=(2 2 4)
for i in "${!pred_lens[@]}"; do
    python run.py \
        --task_name long_term_forecast \
        --seed 5566 \
        --is_training 1 \
        --data_name PEMS07 \
        --model_id PEMS07_96_${pred_lens[$i]} \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len ${pred_lens[$i]} \
        --e_layers ${e_layers[$i]} \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --itr 1 \
        --use_norm 0
done


e_layers=(2 2 4)
for i in "${!pred_lens[@]}"; do
    python run.py \
        --task_name long_term_forecast \
        --seed 5566 \
        --is_training 1 \
        --data_name PEMS08 \
        --model_id PEMS08_96_${pred_lens[$i]} \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --pred_len ${pred_lens[$i]} \
        --e_layers ${e_layers[$i]} \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --itr 1 \
        --use_norm 1
done