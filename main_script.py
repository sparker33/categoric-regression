import pandas as pd
from scipy import rand
from regressor import regressor
from arrange_sim import arrange_sim

data = pd.read_csv("sample_data.tsv", sep='\t')

cardinal_dims_indexes = [2,3]
result_dim_index = 4
base_regression = regressor(data, cardinal_dims_indexes, result_dim_index)
base_regression.print_regression_string()
# base_regression.plot_regression()


data['Base Regression'] = base_regression.get_Y_pred_Series()
data['Base Error'] = data['Base Regression'] - data['Result']

sim = arrange_sim(data, 2, 'Base Error', values_force_const=0.008, ambient_force_const=-0.05,
                    values_order_const=1.0, ambient_order_const=0.0, damping_const=0.1, dist_thresh=10**(-5))
# sim_energy = []
# for i in range(150):
#     sim.run_sim()
#     energy = sim.get_energy()
#     sim_energy.append([i, energy[0], energy[1]])
# sim_energy = pd.DataFrame(sim_energy, columns=['Iteration', 'Potential', 'Kinetic'])

# import matplotlib.pyplot as plt
# plt.plot(sim_energy['Iteration'], sim_energy['Potential'])
# plt.plot(sim_energy['Iteration'], sim_energy['Kinetic'])
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
fig = plt.figure()
ax = fig.add_axes([0,0,1,1], frameon=False)
ax.set_xlim(0,1), ax.set_xticks([])
ax.set_ylim(0,1), ax.set_yticks([])
scat = ax.scatter(x=[a.pos[0] for a in sim.agents], y=[a.pos[1] for a in sim.agents], c=[a.val for a in sim.agents], cmap='Greens')
def update(frame):
    sim.run_sim(duration=0.1, increment=0.1)
    scat.set_offsets([a.pos for a in sim.agents])
    return

anim = animation.FuncAnimation(fig, update, interval=100)
plt.show()