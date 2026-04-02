model_name=PatchMLP
dataset=Illness
model_id=Illness_PatchMLP
model_config=configs/models/Illness/PatchMLP.yaml

pred_lens=(24 36 48 60)
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
      --seq_len 36 \
      --label_len 18 \
      --pred_len ${pred_len} \
      --model_config $model_config \
      --des 'Exp' \
      --itr 1 
done
