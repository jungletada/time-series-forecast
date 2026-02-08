dataset=Electricity_dep
model_id=NDA+TimeFilter
model_name=TimeFilter
d_model=(512 512 512 512)
d_ff=(512 512 512 512)
dropout=(0.5 0.4 0.4 0.4)
patch_len=(32 32 32 32)
pred_lens=(96 192 336 720)


model_configs=(
    configs/models/Electricity/Timefilter_0.yaml 
    configs/models/Electricity/Timefilter_1.yaml 
    configs/models/Electricity/Timefilter_2.yaml)

pivot=4
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
    --task_name long_term_forecast \
    --model_configs ${model_configs[@]} \
    --model $model_name \
    --data_name $dataset \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --pivot $pivot \
    --des 'Exp' \
    --itr 1
done