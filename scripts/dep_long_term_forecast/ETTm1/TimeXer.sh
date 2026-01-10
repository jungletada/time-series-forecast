export CUDA_VISIBLE_DEVICES=0
features=S
model_name=TimeXer
data_name=ETTm1_dep
d_model=(256 256 256 256)
d_ff=(2048 256 1024 512)
batch_size=(4 4 4 4)
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
    --d_model ${d_model[$i]} \
    --d_ff ${d_ff[$i]} \
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
    --d_model ${d_model[$i]} \
    --d_ff ${d_ff[$i]} \
    --batch_size ${batch_size[$i]} \
    --itr 1
done