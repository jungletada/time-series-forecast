#!/bin/bash
mnn_type=mlp

# Long-term forecasting scripts
all_scripts=(
    # scripts/dep_long_term_forecast/Electricity/TimeXer.sh
    # scripts/dep_long_term_forecast/ETTh1/TimeXer.sh
    scripts/dep_long_term_forecast/ETTh2/TimeXer.sh
    scripts/dep_long_term_forecast/ETTm1/TimeXer.sh
    # scripts/dep_long_term_forecast/ETTm2/TimeXer.sh
    # scripts/dep_long_term_forecast/Exchange/TimeXer.sh
    scripts/dep_long_term_forecast/Illness/TimeXer.sh
    scripts/dep_long_term_forecast/Weather/TimeXer.sh
    # scripts/dep_long_term_forecast/Traffic/TimeXer.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done