model_name=TimeMixer
model_id=NDA+TimeMixer
dataset=Traffic_dep
seq_len=96
# Hyperparameters
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64


pred_lens=(720)
model_configs=(configs/models/Traffic/TimeMixer_0.yaml configs/models/Traffic/TimeMixer_1.yaml configs/models/Traffic/TimeMixer_2.yaml)
for i in "${!pred_lens[@]}"; do
    # python -u run_dep.py \
    #     --task_name long_term_forecast \
    #     --is_training 1 \
    #     --data_name $dataset \
    #     --model_id $model_id \
    #     --model $model_name \
    #     --model_configs ${model_configs[@]} \
    #     --features S \
    #     --seq_len $seq_len \
    #     --label_len 0 \
    #     --pred_len ${pred_lens[$i]} \
    #     --batch_size 8 \
    #     --train_epochs 5 \
    #     --patience 10 \
    #     --des 'Exp' \
    #     --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model_id $model_id \
        --model $model_name \
        --model_configs ${model_configs[@]} \
        --features S \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --batch_size 16 \
        --des 'Exp' \
        --itr 1
done