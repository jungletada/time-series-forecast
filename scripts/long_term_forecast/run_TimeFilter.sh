#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/long_term_forecast/ETT_script/TimeFilter_ETTh1.sh
    scripts/long_term_forecast/ETT_script/TimeFilter_ETTh2.sh
    scripts/long_term_forecast/ETT_script/TimeFilter_ETTm1.sh
    scripts/long_term_forecast/ETT_script/TimeFilter_ETTm2.sh
    scripts/long_term_forecast/ILI_script/TimeFilter.sh
    scripts/long_term_forecast/Exchange_script/TimeFilter.sh
    scripts/long_term_forecast/Traffic_script/TimeFilter.sh
    scripts/long_term_forecast/Weather_script/TimeFilter.sh
    scripts/long_term_forecast/ECL_script/TimeFilter.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done