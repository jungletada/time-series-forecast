dataset=Electricity_dep
model_id=NDA+TimeFilter
model_name=TimeFilter
d_model=(512 512 512 512)
d_ff=(512 512 512 512)
dropout=(0.5 0.4 0.4 0.4)
patch_len=(32 32 32 32)

model_configs=(configs/models/Electricity/Timefilter_0.yaml configs/models/Electricity/Timefilter_1.yaml configs/models/Electricity/Timefilter_2.yaml)
pred_lens=(720)

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id $model_id \
    --model $model_name \
    --model_configs ${model_configs[@]} \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --des 'Exp' \
    --batch_size 16 \
    --train_epochs 12 \
    --patience 10 \
    --print_freq 10 \
    --itr 1 \


    # python -u run_dep.py \
    # --task_name long_term_forecast \
    # --is_training 0 \
    # --use_mnn 1 \
    # --mnn $mnn \
    # --data_name $dataset \
    # --model_id $model_id \
    # --model $model_name \
    # --features S \
    # --seq_len 96 \
    # --label_len 48 \
    # --pred_len ${pred_lens[$i]} \
    # --e_layers 2 \
    # --d_layers 1 \
    # --factor 3 \
    # --patch_len ${patch_len[$i]} \
    # --des 'Exp' \
    # --learning_rate 0.001 \
    # --batch_size 32 \
    # --train_epochs 15 \
    # --d_model ${d_model[$i]}\
    # --d_ff ${d_ff[$i]}\
    # --dropout ${dropout[$i]} \
    # --itr 1 \
    # --inverse
done
