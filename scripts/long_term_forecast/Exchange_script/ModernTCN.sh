model_name=ModernTCN
dataset=Exchange
pred_lens=(96)
model_config=configs/models/Exchange/ModernTCN.yaml

for i in "${!pred_lens[@]}"; do
  python -u run.py \
    --seed 3407 \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id $model_name \
    --model $model_name \
    --model_config $model_config \
    --pred_len ${pred_lens[$i]} \
    --seq_len 6 \
    --features S \
    --target OT 
done

