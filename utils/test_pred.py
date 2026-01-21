import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.metrics import metric

path1 = 'dataset/PEMS/gt_merged_PEMS03_test_sl96_cd.npy'
path2 = 'dataset/PEMS/gt_merged2_PEMS03_test_sl96_cd.npy'
path3 = 'dataset/PEMS/PEMS03_test_sl96_cd.npy'

 # Metrics Calculation
preds1=np.load(path1)
preds2 = np.load(path2)
trues = np.load(path3)
print(preds1.shape, preds2.shape, trues.shape)
mae1, mse1, rmse1, mape1, mspe1 = metric(preds1, trues)
mae2, mse2, rmse2, mape2, mspe2 = metric(preds2, trues)
print(f"MAE1: {mae1}, MSE1: {mse1}, RMSE1: {rmse1}, MAPE1: {mape1}, MSPE1: {mspe1}")
print(f"MAE2: {mae2}, MSE2: {mse2}, RMSE2: {rmse2}, MAPE2: {mape2}, MSPE2: {mspe2}")

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