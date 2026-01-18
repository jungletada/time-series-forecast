model_name=RNN
rnn_type=RNN
d_model=64
e_layers=3
dropout=0.1
features=S
pred_lens=(12 24 48)
# datasets=(PEMS03_dep PEMS04_dep PEMS07_dep PEMS08_dep)

dataset=PEMS03_dep
for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 32 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --patience 5 \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1
done

dataset=PEMS04_dep
for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 32 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --patience 5 \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1
done


dataset=PEMS07_dep
for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 32 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --patience 5 \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1
done


dataset=PEMS08_dep
for pred_len in "${pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 32 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --patience 5 \
        --itr 1

    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --data_name $dataset \
        --model $model_name \
        --features $features \
        --seq_len 96 \
        --label_len 0 \
        --pred_len $pred_len \
        --rnn_type $rnn_type \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1
done

