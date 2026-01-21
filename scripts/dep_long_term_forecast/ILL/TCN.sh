d_model=32
e_layers=4
features=S
data_name=Illness_dep
model_id=NDA+TCN    
model_name=TCN

pred_lens=(24)
model_configs=(configs/models/Illness/TCN_0.yaml configs/models/Illness/TCN_1.yaml configs/models/Illness/TCN_2.yaml)
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $data_name \
        --model_id $model_id \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --features $features \
        --seq_len 36 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 18
        
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name Illness_dep \
        --model_id $model_id \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --features $features \
        --seq_len 36 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --batch_size 40 \
        --des 'Exp' \
        --itr 1  
done
