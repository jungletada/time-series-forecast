#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/long_term_forecast/Weather_script/TimeMixer.sh
    scripts/long_term_forecast/Weather_script/iTransformer.sh
    scripts/long_term_forecast/Weather_script/Autoformer.sh
    scripts/long_term_forecast/Weather_script/Crossformer.sh
    scripts/long_term_forecast/Weather_script/DLinear.sh
    scripts/long_term_forecast/Weather_script/TimesNet.sh
    scripts/long_term_forecast/Weather_script/PatchTST.sh
    scripts/long_term_forecast/Weather_script/PAttn.sh
    scripts/long_term_forecast/Weather_script/MultiPatchFormer.sh
    scripts/long_term_forecast/Weather_script/TimeXer.sh
    scripts/long_term_forecast/Weather_script/WPMixer.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done