import pandas as pd
import numpy as np

from regressor import regressor
from arrange_sim import arrange_sim
from category_mapper import category_mapper

data = pd.read_csv("sample_data.tsv", sep='\t')

cardinal_dims_indexes = [2,3]
result_dim_index = 4
base_regression = regressor(data, cardinal_dims_indexes, result_dim_index)
# base_regression.print_regression_string()
# base_regression.plot_regression()

data['Base Regression'] = base_regression.get_Y_pred_Series()
result_norm = sum([abs(r) for r in data['Result']]) / len(data['Result'])
data['Base Error'] = (data['Base Regression'] - data['Result']) / result_norm

category_dims = ['Category 1', 'Category 2']
vals_dim = 'Base Error'
sim = arrange_sim(data, category_dims, vals_dim, values_force_const=0.012, categories_force_const=0.1, ambient_force_const=-0.2,
                    values_order_const=1.0, categories_order_const=0.0, ambient_order_const=0.0, damping_const=0.33, dist_thresh=10**(-8))

################
#### PROTOTYPING PLOTS 1
################
import matplotlib.pyplot as plt
import matplotlib.animation as animation
fig,axs = plt.subplots(2,2)
ID1_colors = {'A':1, 'B':2, 'C':3, 'D':4, 'Z':5}
ID2_colors = {'ID1':1, 'ID2':2, 'ID3':3, 'ID4':4, 'ID5':5}
scat_vals = axs[0,0].scatter(x=[a.pos[0] for a in sim.agents], y=[a.pos[1] for a in sim.agents], c=[a.val for a in sim.agents], cmap='YlGnBu')
scat_cat1 = axs[0,1].scatter(x=[a.pos[0] for a in sim.agents], y=[a.pos[1] for a in sim.agents], c=[ID1_colors[id] for id in list(data['Category 1'])], cmap='Spectral')
scat_cat2 = axs[1,0].scatter(x=[a.pos[0] for a in sim.agents], y=[a.pos[1] for a in sim.agents], c=[ID2_colors[id] for id in list(data['Category 2'])], cmap='Spectral')
times = [0.0]
sim_energy = [[0.0], [0.0]]
lines = [axs[1,1].plot([], [])[0] for _ in sim_energy]
def update(frame, lines):
    sim.run_sim(duration=0.1, increment=0.1)
    scat_vals.set_offsets([[a.pos[0],a.pos[1]] for a in sim.agents])
    scat_cat1.set_offsets([[a.pos[0],a.pos[1]] for a in sim.agents])
    scat_cat2.set_offsets([[a.pos[0],a.pos[1]] for a in sim.agents])
    energy = sim.get_energy()
    times.append(times[-1]+0.1)
    axs[1,1].set_xlim(0.0,times[-1])
    axs[1,1].set_ylim(0.0,np.max([np.max(sim_energy[0]), np.max(sim_energy[1]), 0.1]))
    for i,line in enumerate(lines):
        sim_energy[i].append(energy[i])
        line.set_data(np.array([list(te) for te in zip(times, sim_energy[i])]).T)
    return
anim = animation.FuncAnimation(fig, update, fargs=(lines,), interval=50)
plt.show()
################
#### END PROTOTYPING PLOTS 1
################

point_positions = sim.get_positions_df()
arranged_points_norm = category_mapper.orient_points(point_positions)
mapper = category_mapper.get_mapper(data, arranged_points_norm, category_dims)

################
#### PROTOTYPING PLOTS 2
################
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