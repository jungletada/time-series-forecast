export CUDA_VISIBLE_DEVICES=0
# For MS, M:
#   --enc_in 7 \ 
#   --dec_in 7 \  
#   --c_out 7 \   

pred_lens=(24 36 48 60)
for i in "${!pred_lens[@]}"; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_name Illness \
  --model_id Ill_36_${pred_lens[$i]} \
  --model iTransformer \
  --features S \
  --target OT \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --seq_len 36 \
  --label_len 18 \
  --pred_len ${pred_lens[$i]} \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1
done