model_name=TimeFilter
seq_len=96
features=MS
pred_lens=(12 24 48)

for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS03 \
    --model_id PEMS03_96_${pred_lens[$i]} \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --patch_len 48 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --dropout 0.1 \
    --top_p 0.0 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 20 \
    --itr 1
done

for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS04 \
    --model_id PEMS04_96_${pred_lens[$i]} \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
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


for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS07 \
    --model_id PEMS07_96_${pred_lens[$i]} \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --patch_len 96 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 1024 \
    --dropout 0.1 \
    --top_p 0.0 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 20 \
    --itr 1 \
    --use_norm 0
done

for i in "${!pred_lens[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name PEMS08 \
    --model_id PEMS08_96_${pred_lens[$i]} \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
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
