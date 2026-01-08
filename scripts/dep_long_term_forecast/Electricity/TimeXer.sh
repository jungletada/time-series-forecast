export CUDA_VISIBLE_DEVICES=0
# For MS, M:
#   --enc_in 321 \ 
#   --dec_in 321 \  
#   --c_out 321 \ 

pred_lens=(96 192 336 720)
for i in "${!pred_lens[@]}"; do 
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name Electricity_dep \
    --model_id ECL_dep_96_${pred_lens[$i]} \
    --model TimeXer \
    --features S \
    --target OT \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 4 \
    --factor 3 \
    --des 'Exp' \
    --d_ff 512 \
    --batch_size 4 \
    --itr 1

    python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --mnn mlp \
    --data_name Electricity_dep \
    --model_id ECL_dep_96_${pred_lens[$i]} \
    --model TimeXer \
    --features S \
    --target OT \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
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