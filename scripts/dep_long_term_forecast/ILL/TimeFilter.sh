model_name=TimeFilter
model_id=NDA+TimeFilter
d_model=(128 128 128 128)
d_ff=(512 512 512 512)
dropout=(0.3 0.3 0.3 0.3)
patch_len=(36 36 36 36)
data_name=Illness_dep
model_configs=(configs/models/Illness/Timefilter_0.yaml configs/models/Illness/Timefilter_1.yaml configs/models/Illness/Timefilter_2.yaml)


pred_lens=(24)
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $data_name \
        --model $model_name \
        --model_id $model_id \
        --model_configs ${model_configs[@]} \
        --features S \
        --seq_len 36 \
        --label_len 18 \
        --pred_len ${pred_lens[$i]} \
        --batch_size 32 \
        --train_epochs 10 \
        --des 'Exp' \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $data_name \
        --model $model_name \
        --model_id $model_id \
        --model_configs ${model_configs[@]} \
        --features S \
        --seq_len 36 \
        --label_len 18 \
        --pred_len ${pred_lens[$i]} \
        --batch_size 32 \
        --train_epochs 10 \
        --des 'Exp' \
        --itr 1
done
