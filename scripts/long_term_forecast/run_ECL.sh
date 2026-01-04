#!/bin/bash

# Long-term forecasting scripts
all_scripts=(
    scripts/long_term_forecast/ECL_script/TimeMixer.sh
    scripts/long_term_forecast/ECL_script/iTransformer.sh
    scripts/long_term_forecast/ECL_script/Autoformer.sh
    scripts/long_term_forecast/ECL_script/Crossformer.sh
    scripts/long_term_forecast/ECL_script/DLinear.sh
    scripts/long_term_forecast/ECL_script/TimesNet.sh
    scripts/long_term_forecast/ECL_script/PatchTST.sh
    scripts/long_term_forecast/ECL_script/PAttn.sh
    scripts/long_term_forecast/ECL_script/MultiPatchFormer.sh
    scripts/long_term_forecast/ECL_script/TimeXer.sh
    scripts/long_term_forecast/ECL_script/WPMixer.sh
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done
