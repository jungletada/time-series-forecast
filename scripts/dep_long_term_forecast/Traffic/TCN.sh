model_name=TCN
model_id=NDA+TCN
d_model=32
e_layers=4
pred_lens=(720)

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --data_name Traffic_dep \
        --model_id $model_id \
        --model $model_name \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --features S \
        --des 'Exp' \
        --itr 1 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 64

    # python -u run_dep.py \
    #     --task_name long_term_forecast \
    #     --is_training 0 \
    #     --use_mnn 1 \
    #     --mnn resmlp \
    #     --data_name Traffic_dep \
    #     --model_id Traffic_96_${pred_lens[$i]} \
    #     --model TCN \
    #     --seq_len 96 \
    #     --label_len 0 \
    #     --pred_len ${pred_lens[$i]} \
    #     --enc_in 1 \
    #     --c_out 1 \
    #     --target OT \
    #     --features S \
    #     --d_model $d_model \
    #     --e_layers $e_layers \
    #     --des 'Exp' \
    #     --itr 1 
done