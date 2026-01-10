#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/dep_long_term_forecast/Electricity/Transformer.sh
    scripts/dep_long_term_forecast/ETTh1/Transformer.sh
    scripts/dep_long_term_forecast/ETTh2/Transformer.sh
    scripts/dep_long_term_forecast/ETTm1/Transformer.sh
    scripts/dep_long_term_forecast/ETTm2/Transformer.sh
    scripts/dep_long_term_forecast/Exchange/Transformer.sh
    scripts/dep_long_term_forecast/Illness/Transformer.sh
    scripts/dep_long_term_forecast/Weather/Transformer.sh
    scripts/dep_long_term_forecast/Traffic/Transformer.sh
    scripts/dep_short_term_forecast/PEMS/Transformer.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done