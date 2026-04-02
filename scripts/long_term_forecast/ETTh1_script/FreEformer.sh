dataset=ETTh1
model_id=FreEformer_ETTh1
model_name=FrePatchTST3_attn_ablation
model_config=configs/models/ETTh1/FreEformer.yaml

pred_lens=(96)

for pred_len in "${pred_lens[@]}"; do
    python -u run.py \
      --is_training 0 \
      --seed 3407 \
      --task_name long_term_forecast \
      --data_name $dataset \
      --model_id $model_id \
      --model $model_name \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --model_config $model_config \
      --des 'Exp'
done
