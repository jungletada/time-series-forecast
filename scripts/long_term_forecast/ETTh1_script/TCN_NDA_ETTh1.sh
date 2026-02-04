model_name=TCN_NDA  
dataset=ETTh1
pred_lens=(96 192 336 720)
d_models=(32 64 96 96)
e_layers=(4 4 4 4)
kernel_size=(5 7 9 9)
dropouts=(0.3 0.3 0.3 0.3)
learning_rates=(0.0002 0.002 0.002 0.002)

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --seed 3407 \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id $model_name \
    --model $model_name \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --d_model ${d_models[$i]} \
    --e_layers ${e_layers[$i]} \
    --kernel_size ${kernel_size[$i]} \
    --dropout ${dropouts[$i]} \
    --learning_rate ${learning_rates[$i]} \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --print_freq 100 \
    --des 'Exp' \
    --itr 1
done
