model_name=TimeFilter
d_model=512
e_layers=2
dropout=0.1
features=S
pred_lens=(12 24 48)
d_ffs=(512 512 1024 512)
patch_lens=(48 48 96 48)
use_norm=(1 0 0 1)
datasets=(PEMS03_dep PEMS04_dep PEMS07_dep PEMS08_dep)

for j in "${!datasets[@]}"; do
    dataset=${datasets[$j]}
    for i in "${!pred_lens[@]}"; do
        python -u run_dep.py \
            --task_name long_term_forecast \
            --seed 5566 \
            --is_training 1 \
            --data_name $dataset \
            --model $model_name \
            --features $features \
            --seq_len 96 \
            --pred_len ${pred_lens[$i]} \
            --d_model $d_model \
            --e_layers $e_layers \
            --patch_len ${patch_lens[$j]} \
            --d_ff ${d_ffs[$j]} \
            --dropout $dropout \
            --top_p 0.0 \
            --des 'Exp' \
            --batch_size 16 \
            --learning_rate 0.001 \
            --train_epochs 20 \
            --patience 10 \
            --use_norm ${use_norm[$j]} \
            --itr 1

        python -u run_dep.py \
            --task_name long_term_forecast \
            --seed 5566 \
            --is_training 0 \
            --use_mnn 1 \
            --mnn mlp \
            --data_name $dataset \
            --model $model_name \
            --features $features \
            --seq_len 96 \
            --pred_len ${pred_lens[$i]} \
            --patch_len ${patch_lens[$j]} \
            --d_model $d_model \
            --e_layers $e_layers \
            --patch_len ${patch_lens[$j]} \
            --d_ff ${d_ffs[$j]} \
            --dropout $dropout \
            --top_p 0.0 \
            --des 'Exp' \
            --batch_size 16 \
            --use_norm ${use_norm[$j]} \
            --itr 1
    done
done
