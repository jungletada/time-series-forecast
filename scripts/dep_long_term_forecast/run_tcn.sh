#!/bin/bash
mnn_type=mlp

# Long-term forecasting scripts
all_scripts=(
    scripts/dep_long_term_forecast/Electricity/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/ETTh1/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/ETTh2/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/ETTm1/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/ETTm2/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Exchange/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Illness/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Weather/TCN.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Traffic/TCN.sh --mnn $mnn_type
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done