import numpy as np

class agent_point:
    def __init__(self, id, val:float, dims:int, mass:float = 1.0):
        self.id = id
        self.val = val
        self.force = np.zeros(dims)
        self.accel = np.zeros(dims)
        self.vel = np.zeros(dims)
        self.pos = np.random.rand(dims)
        self.mass = mass
        return

    def clear_forces(self) -> None:
        self.force = np.zeros(len(self.pos))
        return

    def update_kinematics(self, dt:float, damping:float = 0.05) -> None:
        assert self.mass != 0.0
        accel = self.force / self.mass     # a1 = a0 + f/m
        self.vel = self.vel * (1.0 - damping)   # apply damping to v0
        self.pos = self.pos + self.vel * dt + 0.25 * (self.accel + accel) * dt * dt    # x1 = x0 + v0*t + 0.5*a_avg*t^2
        self.vel = self.vel + 0.5*(self.accel + accel) * dt     # v1 = v0 + a_avg*t
        self.accel = accel
        return

    def update_forces(self, force_increment:np.array) -> None:
        self.force += force_increment
        return

    def calculate_forces(self, other_agent, c_vals:float, c_ambient:float, o_vals:float = 1.0, o_ambient:float = 0.0, dist_thresh:float = 10**(-10)) -> np.array:
        assert len(self.pos) == len(other_agent.pos)
        assert type(other_agent) == type(self)
        direction = self.pos - other_agent.pos
        distance = np.linalg.norm(direction)
        force_increment = np.zeros(len(direction))
        if distance > dist_thresh:
            force_increment = (c_vals * abs(self.val - other_agent.val) / (distance**o_vals)) * direction
        center_direction = self.pos - 0.5*np.ones(len(self.pos))
        center_distance = np.linalg.norm(center_direction)
        if center_distance > dist_thresh:
            force_increment += (c_ambient / (center_distance**o_ambient)) * center_direction
        return force_increment

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