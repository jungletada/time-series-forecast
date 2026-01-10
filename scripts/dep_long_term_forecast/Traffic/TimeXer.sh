export CUDA_VISIBLE_DEVICES=0

pred_lens=(96 192 336 720)
data_name=Traffic_dep
model_name=TimeXer
e_layers=3
d_model=512
d_ff=512
features=S

for i in "${!pred_lens[@]}"; do 
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $data_name \
    --model_id $data_name_96_${pred_lens[$i]} \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --factor 3 \
    --des 'Exp' \
    --d_ff $d_ff \
    --d_model $d_model \
    --batch_size 4 \
    --itr 1

    python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --mnn mlp \
    --data_name $data_name \
    --model_id $data_name_96_${pred_lens[$i]} \
    --model $model_name \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers $e_layers \
    --factor 3 \
    --des 'Exp' \
    --d_ff $d_ff \
    --d_model $d_model \
    --batch_size 4 \
    --itr 1
done