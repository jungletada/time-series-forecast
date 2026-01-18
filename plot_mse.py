import re
import matplotlib.pyplot as plt

log_file = "results/Illness/long_term_forecast_Illness_dep_TCN_seq36_pred24_ft(S)_#0/run.log"
mse_values = []

print(f"Reading from {log_file}...")

try:
    with open(log_file, 'r') as f:
        for line in f:
            # Look for "mse:0.xxxx" pattern
            match = re.search(r'mse:(\d+\.\d+)', line)
            if match:
                mse_values.append(float(match.group(1)))
except FileNotFoundError:
    print(f"Error: File {log_file} not found.")
    exit(1)

print(f"Extracted {len(mse_values)} MSE values: {mse_values}")

if not mse_values:
    print("No MSE values found in the log file.")
    exit(0)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(mse_values)), mse_values, color='skyblue', edgecolor='black')
plt.xlabel('Index')
plt.ylabel('MSE')
plt.title('MSE Values from Log File')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set x-axis ticks to be integers
plt.xticks(range(len(mse_values)))

# Add text labels on top of bars
y_max = max(mse_values)
y_min = min(mse_values)
y_range = y_max - y_min
plt.ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range) # Adjust y-limits for better visualization

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', ha='center', va='bottom', rotation=45, fontsize=8)

output_file = "mse_plot.png"
plt.tight_layout()
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
