
# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.000913333 0.001379042 0.000607991 0.001470479)
batches=(32 64 32 128)
epochs=(60 60 60 60)
dropouts=(0.4 0.4 0.4 0.4)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(256 128 128 128)
patiences=(12 12 12 12)

# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(db2 db3 db2 db2)
levels=(2 2 1 1)
tfactors=(5 5 3 5)
dfactors=(8 5 3 3)
strides=(8 8 8 8)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run_dep.py \
		--task_name long_term_forecast \
		--data_name Weather_dep \
		--is_training 1 \
		--use_amp \
		--model_id Weather_dep_${seq_lens[$i]}_${pred_lens[$i]} \
		--model WPMixer \
		--features S \
    	--target OT \
    	--c_out 1 \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--label_len 0 \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--dropout ${dropouts[$i]}
done


for i in "${!pred_lens[@]}"; do
	python -u run_dep.py \
		--task_name long_term_forecast \
		--data_name Weather_dep \
		--is_training 0 \
		--use_mnn 1 \
		--use_amp \
		--model_id Weather_dep_${seq_lens[$i]}_${pred_lens[$i]} \
		--model WPMixer \
		--features S \
    	--target OT \
    	--c_out 1 \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--label_len 0 \
		--batch_size ${batches[$i]} \
		--lradj ${lradjs[$i]} \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--dropout ${dropouts[$i]}
done
