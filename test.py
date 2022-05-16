import pandas as pd
import numpy as np
from scipy import optimize

from regressor import regressor
from arrange_sim import arrange_sim

data = pd.read_csv("sample_data.tsv", sep='\t')

cardinal_dims_indexes = [2,3]
result_dim_index = 4
base_regression = regressor(data, cardinal_dims_indexes, result_dim_index)
base_regression.print_regression_string()
# base_regression.plot_regression()

data['Base Regression'] = base_regression.get_Y_pred_Series()
result_norm = sum([abs(r) for r in data['Result']]) / len(data['Result'])
data['Base Error'] = (data['Base Regression'] - data['Result']) / result_norm

category_dims = ['Category 1', 'Category 2']
vals_dim = 'Base Error'
sim = arrange_sim(data, category_dims, vals_dim, values_force_const=0.012, categoriees_force_const=0.1, ambient_force_const=-0.2,
                    values_order_const=1.0, categories_order_const=0.0, ambient_order_const=0.0, damping_const=0.33, dist_thresh=10**(-8))


point_positions = sim.get_positions_df()

test_df = pd.merge(data, point_positions, left_index=True, right_index=True)

test_agg = test_df.groupby('Category 1').agg({'Result': 'mean'})
for a in test_agg.iterrows():
    print(a[0])


# arr_test = []
# arr_test.append([1,2,3])
# arr_test.append([4,8,6])
# arr_test.append([7,1,9])
# # arr_test = np.array(arr_test)
# r,c = optimize.linear_sum_assignment(arr_test)
# print(r)
# print(c)
# # print(arr_test[r,c])