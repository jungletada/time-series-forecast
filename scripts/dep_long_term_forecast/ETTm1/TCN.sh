export CUDA_VISIBLE_DEVICES=0
dataset=ETTm1_dep
d_model=16
e_layers=4
pred_lens=(96 192 336 720)

mnn="mlp"
if [[ $# -gt 0 ]]; then
  case "$1" in
    --mnn)
      mnn="${2:?Missing value for --mnn}"
      ;;
    *)
      echo "Usage: $0 [--mnn NAME]" >&2
      exit 1
      ;;
  esac
fi

for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --data_name $dataset \
        --model TCN \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --enc_in 1 \
        --c_out 1 \
        --target OT \
        --features S \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.005 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 128
done

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --mnn $mnn \
        --data_name $dataset \
        --model TCN \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --features S \
        --target OT \
        --enc_in 1 \
        --c_out 1 \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1
done
