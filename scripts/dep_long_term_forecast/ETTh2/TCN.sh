model_name=TCN
model_id=NDA+TCN
dataset=ETTh2_dep
seq_len=96
d_model=32
e_layers=4
pred_lens=(96 192 336 720)
model_configs=(configs/models/ETTh2/TCN_0.yaml configs/models/ETTh2/TCN_1.yaml configs/models/ETTh2/TCN_2.yaml)

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model_configs ${model_configs[@]} \
        --model $model_name \
        --model_id $model_id \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --features S \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 32

    # python -u run_dep.py \
    #     --task_name long_term_forecast \
    #     --is_training 0 \
    #     --use_mnn 1 \
    #     --model_configs ${model_configs[@]} \
    #     --data_name $dataset \
    #     --model $model_name \
    #     --model_id $model_id \
    #     --seq_len $seq_len \
    #     --label_len 0 \
    #     --pred_len ${pred_lens[$i]} \
    #     --features S \
    #     --des 'Exp' \
    #     --itr 1
done
