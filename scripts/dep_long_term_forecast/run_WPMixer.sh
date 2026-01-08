#!/bin/bash
mnn_type=mlp

# Long-term forecasting scripts
all_scripts=(
    # scripts/dep_long_term_forecast/Electricity/WPMixer.sh --mnn $mnn_type
    # scripts/dep_long_term_forecast/ETTh1/WPMixer.sh --mnn $mnn_type
    # scripts/dep_long_term_forecast/ETTh2/WPMixer.sh --mnn $mnn_type
    # scripts/dep_long_term_forecast/ETTm1/WPMixer.sh --mnn $mnn_type
    # scripts/dep_long_term_forecast/ETTm2/WPMixer.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Exchange/WPMixer.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Illness/WPMixer.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Weather/WPMixer.sh --mnn $mnn_type
    scripts/dep_long_term_forecast/Traffic/WPMixer.sh --mnn $mnn_type
)

for script in "${all_scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "Warning: Script $script not found, skipping."
    fi
done