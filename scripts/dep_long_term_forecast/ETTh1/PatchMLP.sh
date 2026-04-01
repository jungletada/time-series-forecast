model_name=PatchMLP
dataset=ETTh1_dep
model_id=ETTh1_PatchMLP
model_configs=(
    configs/models/ETTh1/PatchMLP_0.yaml 
    configs/models/ETTh1/PatchMLP_1.yaml 
    configs/models/ETTh1/PatchMLP_2.yaml)

pred_lens=(96 192 336 720)
for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[i]}

    python -u run_dep.py \
      --is_training 1 \
      --task_name long_term_forecast \
      --data_name $dataset \
      --model_id $model_id \
      --model $model_name \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len ${pred_len} \
      --model_configs ${model_configs[@]} \
      --des 'Exp' \
      --itr 1 

    python -u run_dep.py \
      --task_name long_term_forecast \
      --is_training 0 \
      --use_mnn 1 \
      --data_name $dataset \
      --model_id $model_id \
      --model $model_name \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len ${pred_len} \
      --model_configs ${model_configs[@]} \
      --des 'Exp' \
      --itr 1 
done
