import pandas as pd
import numpy as np
from sklearn import preprocessing

cdt_csv = pd.read_csv('../cadet_training_program_matrix.csv', parse_dates=True)
cdt_csv = cdt_csv[(cdt_csv['Rank'] == 'Cdt 1st') | (cdt_csv['Rank'] == 'Cdt')]

date_taken = cdt_csv.drop(['Rank', 'Surname', 'First Name', 'Unit', 'SyllabusCompleted'], axis=1)
date_taken.set_index(['Pnumber'], drop=True, inplace=True)  # 20 x 48 matrix
dt = date_taken.apply(pd.to_datetime, errors='coerce', dayfirst=True)
dn = dt.apply(pd.to_numeric)

# Using my own mind to scale the data
scaled_dates = ((1 / (dn.apply(np.log) - 41)) - 1.1) * 100
sd = (scaled_dates.fillna(0))
# print(scaled_dates[(scaled_dates == scaled_dates.max()) | (scaled_dates == scaled_dates.min())].iloc[-4:]) # 20 x 48 matrix
# print(dt[(dt == dt.max()) | (dt == dt.min())].iloc[-4:])

# Summed as the scaled dates for each pin number
summed_cadetstime = sd.apply(sum, axis=1)  # 20 x 1 vector
# print(summed_cadetstime)
minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 10))
X_minmax_cdt = minmax_scaler.fit_transform(np.array([summed_cadetstime]).T) # 20 x 1 vector
# print(X_minmax.shape)

# Summed as the count of modules completed for each pin number
# summed_cadetsnumber = sd[sd > 0].count(axis=1)  # 20 x 1 vector

# The following will need to be transposed in order to combine
summed_modulestime = 1 / sd.apply(sum, axis=0)  # 48 x 1 vector
# print(summed_modulestime)
X_minmax_mod = minmax_scaler.fit_transform(np.array([summed_modulestime]).T) # 48 x 1 vector
# print(X_minmax_mod)
# summed_modulesnumber = sd[sd > 0].count(axis=0)  # 48 x 1 vector

# matmul cadets and modules to form 20 x 48 matrix.
arr1 = np.array([summed_cadetstime]).T  # 20 x 1
arr2 = np.array([summed_modulestime])# 1 x 48
scaled_arr1 = X_minmax_cdt
scaled_arr2 = X_minmax_mod.T

module_cadet_mat = np.matmul(scaled_arr1, scaled_arr2)  # 20 x 48 matrix
# print(pd.DataFrame(module_cadet_mat).iloc[:2])

ratings = sd * module_cadet_mat
best_modules_sorted = ratings.sum(axis=0).sort_values(ascending=False)
print(best_modules_sorted)