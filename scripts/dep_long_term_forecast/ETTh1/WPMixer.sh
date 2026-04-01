dataset=ETTh1_dep
model_name=WPMixer
model_id=ETTh1_WPMixer
model_configs=(
    configs/models/ETTh1/WPMixer_0.yaml 
    configs/models/ETTh1/WPMixer_1.yaml 
    configs/models/ETTh1/WPMixer_2.yaml)
    
pred_lens=(96 192 336 720)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run_dep.py \
		--task_name long_term_forecast \
		--data_name $dataset \
		--is_training 1 \
		--use_amp \
		--model_id $model_id \
		--model $model_name \
		--model_configs ${model_configs[@]} \
		--features M \
		--seq_len 96 \
		--pred_len ${pred_lens[$i]} \
		--label_len 48 \
		--pivot 3 \
		--des 'Exp' \
		--itr 1

	python -u run_dep.py \
		--task_name long_term_forecast \
		--data_name $dataset \
		--is_training 0 \
		--use_mnn 1 \
		--use_amp \
		--model_id $model_id \
		--model $model_name \
		--model_configs ${model_configs[@]} \
		--features M \
		--seq_len 96 \
		--pred_len ${pred_lens[$i]} \
		--label_len 48 \
		--pivot 3 \
		--des 'Exp' \
		--itr 1
done
