#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    # scripts/dep_long_term_forecast/Electricity/TCN.sh
    scripts/dep_long_term_forecast/ETTh1/TCN.sh
    scripts/dep_long_term_forecast/ETTh2/TCN.sh
    scripts/dep_long_term_forecast/ETTm1/TCN.sh
    scripts/dep_long_term_forecast/ETTm2/TCN.sh
    # scripts/dep_long_term_forecast/Exchange/TCN.sh
    # scripts/dep_long_term_forecast/Illness/TCN.sh
    # scripts/dep_long_term_forecast/Weather/TCN.sh
    # scripts/dep_long_term_forecast/Traffic/TCN.sh
    # scripts/dep_short_term_forecast/PEMS/TCN.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done