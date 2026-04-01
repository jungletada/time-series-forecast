model_name=TimeMixer
model_id=NDA+TimeMixer
dataset=ETTh1_dep
seq_len=96
down_sampling_layers=3
down_sampling_window=2
d_model=16
d_ff=32
e_layers=2
pred_lens=(96 192 336 720)
model_configs=(configs/models/TimeMixer_0_ETTh1.yaml 
              configs/models/TimeMixer_1_ETTh1.yaml 
              configs/models/TimeMixer_2_ETTh1.yaml)

for i in "${!pred_lens[@]}"; do
  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id $model_id \
    --model $model_name \
    --features S \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --model_configs ${model_configs[@]} \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 10 \
    --patience 3 \
    --batch_size 128

  python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --mnn mlp \
    --data_name $dataset \
    --model_id $model_id \
    --model $model_name \
    --features S \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len ${pred_lens[$i]} \
    --model_configs ${model_configs[@]} \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32
done