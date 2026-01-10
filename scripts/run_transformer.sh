#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/long_term_forecast/ETT_script/Transformer.sh
    scripts/long_term_forecast/ILI_script/Transformer.sh
    scripts/long_term_forecast/Exchange_script/Transformer.sh
    scripts/long_term_forecast/Traffic_script/Transformer.sh
    scripts/long_term_forecast/Weather_script/Transformer.sh
    scripts/long_term_forecast/ECL_script/Transformer.sh
    scripts/short_term_forecast/PEMS/Transformer.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done