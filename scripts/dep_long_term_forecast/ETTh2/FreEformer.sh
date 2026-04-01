dataset=ETTh1_dep
model_name=FrePatchTST3_attn_ablation
model_configs=(
    configs/models/ETTh1/FreEformer_0.yaml 
    configs/models/ETTh1/FreEformer_1.yaml 
    configs/models/ETTh1/FreEformer_2.yaml)
    
pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --is_training 1 \
        --task_name long_term_forecast \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --data_name $dataset \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --pivot 3 \
        --des 'Exp' \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --data_name $dataset \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --pivot 3 \
        --des 'Exp' \
        --itr 1
done

