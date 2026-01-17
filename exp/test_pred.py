import numpy as np
path1 = 'results/Electricity/long_term_forecast_Electricity_dep_TCN_seq96_pred96_ft(S)_#0/true.npy'
path2 = 'results/Electricity/long_term_forecast_Electricity_TimeFilter_seq96_pred96_ft(S)_#0/true.npy'

true1 = np.load(path1)
true2 = np.load(path2)

print(true1.shape, true2.shape)
if np.array_equal(true1, true2):
    print("true1 and true2 are exactly equal.")
else:
    print("true1 and true2 are NOT equal.")

print(true1[0, :, -1])
print(true2[0, :, -1])

print(np.mean(true1[0, :, -1]), np.mean(true2[0, :, -1]))
print(np.std(true1[0, :, -1]), np.std(true2[0, :, -1]))