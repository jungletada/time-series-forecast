model_name=PatchMLP
dataset=Weather
model_id=Weather_PatchMLP
model_config=configs/models/Weather/PatchMLP.yaml

pred_lens=(96 192 336 720)
for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[i]}

    python -u run.py \
      --is_training 1 \
      --seed 3407 \
      --task_name long_term_forecast \
      --data_name $dataset \
      --model_id $model_id \
      --model $model_name \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len ${pred_len} \
      --model_config $model_config \
      --des 'Exp' \
      --itr 1 
done
