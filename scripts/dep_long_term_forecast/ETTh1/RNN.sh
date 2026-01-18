model_name=RNN
rnn_type=RNN
model_id=NDA+RNN
features=S
dataset=ETTh1_dep
pred_lens=(96 192 336 720)

# 有作用的参数
d_model=64 
e_layers=3
dropout=0.1

for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --model_id $model_id \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 16 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --patience 5 \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --mnn mlp \
        --data_name $dataset \
        --model $model_name \
        --model_id $model_id \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 32 \
        --itr 1
done
