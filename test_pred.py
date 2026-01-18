import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.metrics import metric

path1 = 'results/Electricity/long_term_forecast_Electricity_dep_TCN_seq96_pred96_ft(S)_#0/pred.npy'
path2 = 'results/Electricity/long_term_forecast_Electricity_TimeFilter_seq96_pred96_ft(S)_#0/true.npy'

 # Metrics Calculation
preds = np.load(path1)
trues = np.load(path2)
mae, mse, rmse, mape, mspe = metric(preds, trues)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, MSPE: {mspe}")
# true1 = np.load(path1)
# true2 = np.load(path2)

# print(true1.shape, true2.shape)
# if np.array_equal(true1, true2):
#     print("true1 and true2 are exactly equal.")
# else:
#     print("true1 and true2 are NOT equal.")

# # If shapes are the same, proceed
# true1_flat = true1.reshape(-1)
# true2_flat = true2.reshape(-1)
# if true1_flat.shape == true2_flat.shape:
#     diff = true1_flat - true2_flat
# else:
#     min_len = min(true1_flat.size, true2_flat.size)
#     true1_flat = true1_flat[:min_len]
#     true2_flat = true2_flat[:min_len]
#     diff = true1_flat - true2_flat

#     print(f"Shapes differ; plotting first {min_len} points. ")
    
# max_diff = np.max(np.abs(diff))
# print(f"Max difference: {max_diff}")

# fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
# axes[0].plot(true1_flat, color='#1E90FF', linewidth=1.8)
# axes[0].set_title('true1')
# axes[1].plot(true2_flat, color='#FF4500', linewidth=1.8)
# axes[1].set_title('true2')
# axes[2].plot(diff, color='#444444', linewidth=1.6)
# axes[2].set_title('diff = true1 - true2')
# plt.tight_layout()
# plt.savefig('true_compare.png', dpi=150)
# plt.close()