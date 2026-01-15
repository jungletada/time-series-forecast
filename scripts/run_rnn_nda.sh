#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/dep_long_term_forecast/Electricity/RNN.sh
    scripts/dep_long_term_forecast/ETTh1/RNN.sh
    scripts/dep_long_term_forecast/ETTh2/RNN.sh
    scripts/dep_long_term_forecast/ETTm1/RNN.sh
    scripts/dep_long_term_forecast/ETTm2/RNN.sh
    scripts/dep_long_term_forecast/Exchange/RNN.sh
    scripts/dep_long_term_forecast/Illness/RNN.sh
    scripts/dep_long_term_forecast/Weather/RNN.sh
    scripts/dep_long_term_forecast/Traffic/RNN.sh
    scripts/dep_short_term_forecast/PEMS/RNN.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done