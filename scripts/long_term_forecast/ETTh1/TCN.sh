model_name=TCN
dataset=ETTh1
pred_lens=(96 192 336 720)
d_models=(64 64 96 96)
e_layers=(4 4 4 4)
kernel_size=(5 7 9 9)
learning_rates=(0.001 0.002 0.002 0.002)

for i in 720; do
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
    --pred_len $i \
    --d_model 64 \
    --e_layers 4 \
    --kernel_size 9 \
    --dropout 0.1 \
    --learning_rate 0.002 \
    --lradj type1 \
    --train_epochs 15 \
    --patience 15 \
    --batch_size 128 \
    --des 'Exp' \
    --itr 1
done
