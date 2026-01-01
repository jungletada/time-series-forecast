
# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTh1
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.000242438 0.000201437 0.000132929 0.000239762)
batches=(256 256 256 256)
epochs=(30 30 30 30)
lradjs=(type3 type3 type3 type3)
patiences=(12 12 12 12)
d_models=(256 256 256 128)
dropouts=(0.4 0.05 0.0 0.2)
patch_lens=(16 16 16 16)

# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(db2 db3 db2 db2)
levels=(2 2 1 1)
tfactors=(5 5 3 5)
dfactors=(8 5 3 3)
strides=(8 8 8 8)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--task_name long_term_forecast \
		--data_name $dataset \
		--is_training 1 \
		--use_amp \
		--model_id wpmixer \
		--model $model_name \
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
