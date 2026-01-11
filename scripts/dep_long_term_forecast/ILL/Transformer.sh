export CUDA_VISIBLE_DEVICES=0
model_name=Transformer
dataset=Illness_dep
pred_lens=(96 192 336 720)
features=S

for pred_len in "${pred_lens[@]}"; do
  python -u run_dep.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 1 \
    --data_name $dataset \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --patience 5 \
    --itr 1

    python -u run_dep.py \
    --task_name long_term_forecast \
    --seed 5566 \
    --is_training 0 \
    --use_mnn 1 \
    --mnn mlp \
    --data_name $dataset \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --des 'Exp' \
    --batch_size 32 \
    --itr 1
done