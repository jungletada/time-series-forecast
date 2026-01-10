export CUDA_VISIBLE_DEVICES=0

pred_lens=(96 192 336 720)
data_name=Weather_dep
features=S
for i in "${!pred_lens[@]}"; do 
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $data_name \
    --model_id Weather_96_${pred_lens[$i]} \
    --model TimeXer \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 4 \
    --factor 3 \
    --des 'Exp' \
    --d_ff 512 \
    --batch_size 4 \
    --itr 1
done

for i in "${!pred_lens[@]}"; do 
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --mnn mlp \
    --data_name $data_name \
    --model_id Weather_96_${pred_lens[$i]} \
    --model TimeXer \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 4 \
    --factor 3 \
    --des 'Exp' \
    --d_ff 512 \
    --batch_size 4 \
    --itr 1
done