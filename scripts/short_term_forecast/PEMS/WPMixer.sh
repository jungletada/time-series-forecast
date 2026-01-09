pred_lens=(12 24 48)
learning_rates=(0.00328086 0.000493286 0.002505375)
batches=(32 32 32)
epochs=(100 100 100)
dropouts=(0.1 0.1 0.2)
patch_lens=(16 16 16)
lradjs=(type3 type3 type3)
d_models=(32 32 32)
patiences=(12 12 12)
features=S

for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--task_name long_term_forecast \
		--is_training 1 \
        --seed 5566 \
		--data_name PEMS03 \
		--model_id PEMS03_96_${pred_lens[$i]} \
		--model WPMixer \
		--features $features \
		--seq_len 96 \
		--label_len 0 \
        --pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp
done

for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--task_name long_term_forecast \
		--is_training 1 \
        --seed 5566 \
		--data_name PEMS04 \
		--model_id PEMS04_96_${pred_lens[$i]} \
		--model WPMixer \
		--features $features \
		--seq_len 96 \
		--label_len 0 \
        --pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp
done

for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--task_name long_term_forecast \
		--is_training 1 \
        --seed 5566 \
		--data_name PEMS07 \
		--model_id PEMS07_96_${pred_lens[$i]} \
		--model WPMixer \
		--features $features \
		--seq_len 96 \
		--label_len 0 \
        --pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp
done

for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--task_name long_term_forecast \
		--is_training 1 \
        --seed 5566 \
		--data_name PEMS08 \
		--model_id PEMS08_96_${pred_lens[$i]} \
		--model WPMixer \
		--features $features \
		--seq_len 96 \
		--label_len 0 \
        --pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp
done