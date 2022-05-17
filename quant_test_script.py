import pandas as pd
import numpy as np
import datetime

from regressor import regressor
from arrange_sim import arrange_sim
from category_mapper import category_mapper

data = pd.read_csv("raw_polls.csv")
data['poll_date_relative'] = data.apply(lambda row: int((datetime.datetime.strptime(row['electiondate'], "%m/%d/%Y") - datetime.datetime.strptime(row['polldate'], "%m/%d/%Y")).days), axis=1)

data_model = data[['location', 'type_detail', 'pollster_rating_id', 'year', 'poll_date_relative', 'cand1_pct', 'cand2_pct', 'cand1_actual', 'cand2_actual']]

cardinal_dims_indexes = [3,4,5,6]
result_dim_index = [7,8]
base_regression = regressor(data_model, cardinal_dims_indexes, result_dim_index)
# base_regression.print_regression_string()
# base_regression.plot_regression()

base_results = base_regression.get_Y_pred_Series()
data_model['Cand1_Base_Pred'] = base_regression.get_Y_pred_Series().apply(lambda val: val[0])
data_model['Cand2_Base_Pred'] = base_regression.get_Y_pred_Series().apply(lambda val: val[1])
data_model['Cand1_Base_Error'] = (data_model['Cand1_Base_Pred'] - data_model['cand1_actual']) / 100.0
data_model['Cand2_Base_Error'] = (data_model['Cand2_Base_Pred'] - data_model['cand2_actual']) / 100.0

category_dims = ['location', 'type_detail', 'pollster_rating_id']
vals_dims = ['Cand1_Base_Error', 'Cand2_Base_Error']
sim = arrange_sim(data_model, category_dims, vals_dims, values_force_const=0.012, categories_force_const=0.1, ambient_force_const=-0.2,
                    values_order_const=1.0, categories_order_const=0.0, ambient_order_const=0.0, damping_const=0.33, dist_thresh=10**(-8))

sim.run_sim(duration=999999, increment=0.1, cutoff_mean_kinetic_energy=10.0**(-8))

point_positions = sim.get_positions_df()
arranged_points_norm = category_mapper.orient_points(point_positions)
mapper = category_mapper.get_mapper(data, arranged_points_norm, category_dims)

################
#### PROTOTYPING PLOTS 2
################
import matplotlib.pyplot as plt
ID1_colors = {'A':1, 'B':2, 'C':3, 'D':4, 'Z':5}
ID2_colors = {'ID1':1, 'ID2':2, 'ID3':3, 'ID4':4, 'ID5':5}
fig, axs = plt.subplots(2,2)
arranged_points = np.array(point_positions)
axs[0,0].scatter(x=arranged_points[:,0], y=arranged_points[:,1], c=[val for val in list(data['Base Error'])], cmap='YlGnBu')
axs[0,1].scatter(x=arranged_points_norm[:,0], y=arranged_points_norm[:,1], c=[val for val in list(data['Base Error'])], cmap='YlGnBu')
axs[1,0].scatter(x=arranged_points_norm[:,0], y=arranged_points_norm[:,1], c=[ID1_colors[id] for id in list(data['Category 1'])], cmap='Spectral')
axs[1,1].scatter(x=arranged_points_norm[:,0], y=arranged_points_norm[:,1], c=[ID2_colors[id] for id in list(data['Category 2'])], cmap='Spectral')
plt.show()
################
#### END PROTOTYPING PLOTS 2
################

from sklearn.linear_model import LinearRegression
data['Category 1 Cardinal'] = data.apply(lambda row: mapper['Category 1'][row['Category 1']], axis=1)
data['Category 2 Cardinal'] = data.apply(lambda row: mapper['Category 2'][row['Category 2']], axis=1)
X = data.iloc[:, [2,3,7,8]].values.reshape(-1, 4)  # values converts it into a numpy array
Y = data.iloc[:, result_dim_index].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
final_linear_regressor = LinearRegression()  # create object for the class
final_linear_regressor.fit(X, Y)  # perform linear regression
final_Y_pred = final_linear_regressor.predict(X)  # make predictions
data['Final Regression'] = pd.Series([p[0] for p in final_Y_pred])
data['Final Error'] = (data['Final Regression'] - data['Result']) / result_norm
print("BaseErr: {}; FinalErr: {}".format(np.linalg.norm(data['Base Error']), np.linalg.norm(data['Final Error'])))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(X[:,2], X[:,3], Y, c=final_Y_pred, cmap='Greens')
# ax.set_xlabel('Category 1')
# ax.set_ylabel('Category 2')
# ax.set_zlabel('Result')
# plt.show()