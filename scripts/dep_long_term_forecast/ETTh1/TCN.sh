export CUDA_VISIBLE_DEVICES=0
model_name=TCN
dataset=ETTh1_dep
seq_len=96
d_model=32
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
        --features S \
        --model $model_name \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1 \
        --learning_rate 0.002 \
        --train_epochs 10 \
        --patience 10 \
        --batch_size 128
    
    python -u run_dep.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --use_mnn 1 \
        --mnn $mnn \
        --data_name $dataset \
        --model $model_name \
        --features S \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len ${pred_lens[$i]} \
        --d_model $d_model \
        --e_layers $e_layers \
        --des 'Exp' \
        --itr 1
done
