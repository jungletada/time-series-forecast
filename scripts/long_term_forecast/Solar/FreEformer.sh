dataset=Solar
model_id=FreEformer_Solar
model_name=FrePatchTST3_attn_ablation
model_config=configs/models/Solar/FreEformer.yaml

pred_lens=(96 )

for pred_len in "${pred_lens[@]}"; do
    python -u run.py \
      --is_training 1 \
      --seed 3407 \
      --task_name long_term_forecast \
      --model_config $model_config \
      --data_name $dataset \
      --model_id $model_id \
      --model $model_name \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --itr 1 \
      --des 'Exp'
done
