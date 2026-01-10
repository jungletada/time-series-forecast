export CUDA_VISIBLE_DEVICES=0
features=S
data_name=ETTh2_dep
model_name=TimeXer
d_model=(256 256 512 256)
d_ff=(1024 1024 1024 1024)
batch_size=(16 16 16 16)
pred_lens=(96 192 336 720)

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
    --e_layers 4 \
    --factor 3 \
    --des 'Exp' \
    --d_ff ${d_ff[$i]} \
    --d_model ${d_model[$i]} \
    --batch_size ${batch_size[$i]} \
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
    --e_layers 4 \
    --factor 3 \
    --des 'Exp' \
    --d_ff ${d_ff[$i]} \
    --d_model ${d_model[$i]} \
    --batch_size ${batch_size[$i]} \
    --itr 1
done