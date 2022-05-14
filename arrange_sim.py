import pandas as pd
import numpy as np
from agent_point import agent_point

class arrange_sim:
    def __init__(self, data: pd.DataFrame, category_dims:list, values_dim,
                values_force_const:float, categoriees_force_const:float, ambient_force_const:float,
                values_order_const:float = 1.0, categories_order_const:float = 0.0, ambient_order_const:float = 0.0, 
                damping_const:float = 0.05, dist_thresh:float = 10**(-10)):
        self.c_vals = values_force_const # positive values repel dissimilar, proportional to difference in values
        self.c_categories = categoriees_force_const # positive values repel dissimilar categories, const
        self.c_ambient = ambient_force_const # positive values repel, const
        self.o_vals = values_order_const # sets order of distance exponential in force calc denominator for values
        self.o_categories = categories_order_const # sets order of distance exponential in force calc denominator for categories
        self.o_ambient = ambient_order_const # sets order of distance exponential in force calc denominator for ambient
        self.damping = damping_const # damping on particle velocities
        self.dist_thresh = dist_thresh # cutoff proximity for force calculations (prevents ~infinite forces)
        self.agents = []
        for row in data.iterrows():
            self.agents.append(agent_point(row[0], row[1][values_dim], row[1][category_dims]))
        return

    def get_positions_df(self) -> pd.DataFrame:
        dimensions = ["X{}".format(i) for i,_ in enumerate(self.agents[0].pos)]
        pos_data = [[d for d in a.pos] for a in self.agents]
        positions = pd.DataFrame(pos_data, columns=dimensions)
        return positions

    def get_energy(self) -> list:
        energy = [0.0, 0.0]
        for a in self.agents:
            energy[0] += a.force@a.force
            energy[1] += 0.5 * a.mass * a.vel@a.vel
        energy[0] = energy[0] / len(self.agents)
        energy[1] = energy[1] / len(self.agents)
        return energy

    def run_sim(self, duration:float = 1.0, increment:float = 0.1, print_prog:bool = False) -> None:
        assert duration > 0.0
        assert increment > 0.0
        t = 0.0
        while t < duration:
            if print_prog:
                print(round(t/duration, 2))
            for a in self.agents:
                a.clear_forces()
            for i,a1 in enumerate(self.agents[:-1]):
                for a2 in self.agents[i+1:]:
                    f = self.calculate_forces(a1, a2)
                    a1.update_forces(f)
                    a2.update_forces(-f)
                a1.update_kinematics(increment, self.damping)
            self.agents[-1].update_kinematics(increment, self.damping)
            t += increment
        return

    def calculate_forces(self, agent1:agent_point, agent2:agent_point) -> np.array:
        assert len(agent1.pos) == len(agent2.pos)
        direction = agent1.pos - agent2.pos
        distance = np.linalg.norm(direction)
        force_magnitude = 0.0
        if distance > self.dist_thresh:
            force_magnitude += ((self.c_vals * abs(agent1.val - agent2.val) / (distance**self.o_vals)) + (self.c_ambient / (distance**self.o_ambient)))
        category_force = (self.c_categories / (distance**self.o_categories))
        for i,cat in enumerate(agent1.categories):
            if agent2.categories[i] != cat:
                force_magnitude += category_force
            else:
                force_magnitude -= category_force
        force_increment = force_magnitude * direction
        return force_increment