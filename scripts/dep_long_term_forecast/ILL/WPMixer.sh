
# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

seq_lens=(36 36 36 36)
pred_lens=(24 36 48 60)
learning_rates=(0.00328086 0.000493286 0.002505375 0.001977516)
batches=(32 32 32 32)
epochs=(100 100 100 100)
dropouts=(0.1 0.1 0.2 0.1)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(32 32 32 32)
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
		--data_name Illness_dep \
		--is_training 1 \
		--use_amp \
		--model_id Ill_dep_${seq_lens[$i]}_${pred_lens[$i]} \
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
		--data_name Illness_dep \
		--is_training 0 \
		--use_mnn 1 \
		--use_amp \
		--model_id Ill_dep_${seq_lens[$i]}_${pred_lens[$i]} \
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
