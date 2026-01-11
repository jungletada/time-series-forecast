export CUDA_VISIBLE_DEVICES=0

pred_lens=(24 36 48 60)
data_name=Illness_dep
model_name=TimeXer
d_ff=512
features=S

for i in "${!pred_lens[@]}"; do 
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $data_name \
    --model $model_name \
    --features $features \
    --seq_len 36 \
    --label_len 18 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 4 \
    --factor 3 \
    --des 'Exp' \
    --d_ff $d_ff \
    --batch_size 4 \
    --itr 1

  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --mnn mlp \
    --data_name $data_name \
    --model $model_name \
    --features $features \
    --seq_len 36 \
    --label_len 18 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 4 \
    --factor 3 \
    --des 'Exp' \
    --d_ff $d_ff \
    --batch_size 4 \
    --itr 1
done
