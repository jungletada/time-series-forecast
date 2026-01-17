model_name=TimeFilter
mnn=mlp
features=S
pred_lens=(96 192 336 720)
d_model=(512 512 512 512)
d_ff=(512 512 512 512)
dropout=(0.5 0.4 0.4 0.4)
patch_len=(32 32 32 32)
dataset=Electricity_dep
model_id=NDA+TimeFilter

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_name $dataset \
    --model_id $model_id \
    --model $model_name \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch_len ${patch_len[$i]} \
    --des 'Exp' \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 15 \
    --d_model ${d_model[$i]}\
    --d_ff ${d_ff[$i]}\
    --dropout ${dropout[$i]} \
    --itr 1 \
    --inverse


    python -u run_dep.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --use_mnn 1 \
    --mnn $mnn \
    --data_name $dataset \
    --model_id $model_id \
    --model $model_name \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_lens[$i]} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --patch_len ${patch_len[$i]} \
    --des 'Exp' \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 15 \
    --d_model ${d_model[$i]}\
    --d_ff ${d_ff[$i]}\
    --dropout ${dropout[$i]} \
    --itr 1 \
    --inverse
done
