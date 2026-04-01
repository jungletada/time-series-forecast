dataset=ETTh1
model_id=ETTh1_freformer_attn1
model_name=FrePatchTST3_attn_ablation
model_config=configs/models/ETTh1/FreEformer.yaml

pred_lens=(96 192 336 720)

for ((i = 0; i < 4; i++))
do
    pred_len=${pred_lens[i]}

    python -u run.py \
      --is_training 1 \
      --is_training 1 \
      --seed 3407 \
      --task_name long_term_forecast \
      --data_name $dataset \
      --model_id $model_id \
      --model $model_name \
      --features M \
      --seq_len 96 \
      --pred_len ${pred_len} \
      --model_config $model_config \
      --des 'Exp'
done
