export CUDA_VISIBLE_DEVICES=0

d_model=36
e_layers=4
pred_lens=(96 192 336 720)
dataset=Weather
# Loop over datasets and prediction lengths
for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id $dataset_96_$pred_len \
    --model TCN \
    --features S \
    --seq_len 96 \
    --label_len 0 \
    --pred_len $pred_len \
    --d_model $d_model \
    --e_layers $e_layers \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.005 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16
done