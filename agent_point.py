import numpy as np

class agent_point:
    def __init__(self, id, val:float, dims:list, mass:float = 1.0):
        self.id = id
        self.val = val
        self.categories = dims
        dim_length = len(dims)+1
        self.force = np.zeros(dim_length)
        self.accel = np.zeros(dim_length)
        self.vel = np.zeros(dim_length)
        self.pos = np.random.rand(dim_length)
        self.mass = mass
        return

    def clear_forces(self) -> None:
        self.force = np.zeros(len(self.pos))
        return

    def update_kinematics(self, dt:float, damping:float = 0.05) -> None:
        assert self.mass != 0.0
        accel = self.force / self.mass     # a1 = a0 + f/m
        self.vel = self.vel * (1.0 - damping)   # apply damping to get v0: v0 = (1-d)*v
        self.pos = self.pos + self.vel * dt + 0.25 * (self.accel + accel) * dt * dt    # x1 = x0 + v0*t + 0.5*a_avg*t^2
        self.vel = self.vel + 0.5*(self.accel + accel) * dt     # v1 = v0 + a_avg*t
        self.accel = accel
        return

    def update_forces(self, force_increment:np.array) -> None:
        self.force += force_increment
        return

#TEST CODE
# agent1 = agent_point(100, 0.2, 2)
# agent2 = agent_point(101, 0.5, 2)
# print(agent1.pos)
# print(agent2.pos)
# f = agent1.calculate_forces(agent2, 0.5, 0.5)
# agent1.update_forces(f)
# agent2.update_forces(-f)
# agent1.update_kinematics(0.1)
# agent2.update_kinematics(0.1)
# print(agent1.pos)
# print(agent2.pos)
# f = agent1.calculate_forces(agent2, 0.5, 0.5)
# agent1.update_forces(f)
# agent2.update_forces(-f)
# agent1.update_kinematics(0.1)
# agent2.update_kinematics(0.1)
# print(agent1.pos)
# print(agent2.pos)