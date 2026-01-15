#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/dep_long_term_forecast/Electricity/DLinear.sh
    scripts/dep_long_term_forecast/ETTh1/DLinear.sh
    scripts/dep_long_term_forecast/ETTh2/DLinear.sh
    scripts/dep_long_term_forecast/ETTm1/DLinear.sh
    scripts/dep_long_term_forecast/ETTm2/DLinear.sh
    scripts/dep_long_term_forecast/Exchange/DLinear.sh
    scripts/dep_long_term_forecast/Illness/DLinear.sh
    scripts/dep_long_term_forecast/Weather/DLinear.sh
    scripts/dep_long_term_forecast/Traffic/DLinear.sh
    scripts/dep_short_term_forecast/PEMS/DLinear.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done