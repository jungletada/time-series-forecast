#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/long_term_forecast/ETT_script/WPMixer_ETTh1.sh
    scripts/long_term_forecast/ETT_script/WPMixer_ETTh2.sh
    scripts/long_term_forecast/ETT_script/WPMixer_ETTm1.sh
    scripts/long_term_forecast/ETT_script/WPMixer_ETTm2.sh
    scripts/long_term_forecast/ECL_script/WPMixer.sh
    scripts/long_term_forecast/Traffic_script/WPMixer.sh
    scripts/long_term_forecast/Weather_script/WPMixer.sh
    scripts/long_term_forecast/Exchange_script/WPMixer.sh
    scripts/long_term_forecast/ILI_script/WPMixer.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done
