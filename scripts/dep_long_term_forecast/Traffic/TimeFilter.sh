data_name=Traffic_dep
model_name=TimeFilter
model_id=NDA+TimeFilter
d_model=(512 512 512 512)
d_ff=(2048 2048 2048 2048)
dropout=(0.3 0.3 0.3 0.3)
patch_len=(96 96 96 96)
e_layers=2
d_layers=1
features=S
factor=3

model_configs=(configs/models/Traffic/Timefilter_0.yaml configs/models/Traffic/Timefilter_1.yaml configs/models/Traffic/Timefilter_2.yaml)
pred_lens=(720)
for i in "${!pred_lens[@]}"; do 
  # python -u run_dep.py \
  #   --task_name long_term_forecast \
  #   --is_training 1 \
  #   --data_name $data_name \
  #   --model_id $model_id \
  #   --model $model_name \
  #   --model_configs ${model_configs[@]} \
  #   --features $features \
  #   --seq_len 96 \
  #   --label_len 48 \
  #   --pred_len ${pred_lens[$i]} \
  #   --des 'Exp' \
  #   --batch_size 16 \
  #   --train_epochs 30 \
  #   --patience 2 \
  #   --print_freq 10 \
  #   --itr 1

  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --data_name $data_name \
    --model_id $model_id \
    --model $model_name \
    --model_configs ${model_configs[@]} \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --des 'Exp' \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 10 \
    --itr 1
done