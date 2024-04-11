import torch
import torch.nn as nn
from torch.nn import functional as F


env_params = {
    "mass": 4.0,
    "inertia": 0.0475,
    "dist": 0.25,
    "gravity": 9.8,
    "dt": 0.05,  # seconds between state updates
    "max_force_ub": 39.2,
    "max_force_lb": 0,
}
STATE_SPACE_SIZE = 6
MID_LAYER_SIZE = 32


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.policy = nn.Linear(STATE_SPACE_SIZE, 2, bias=False)

    def forward(self, x):
        return self.policy(x)


class LyapunovNet(nn.Module):
    def __init__(self):
        super(LyapunovNet, self).__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, MID_LAYER_SIZE, bias=False),
            nn.ReLU(),
            nn.Linear(MID_LAYER_SIZE, MID_LAYER_SIZE, bias=False),
            nn.ReLU(),
            nn.Linear(MID_LAYER_SIZE, 1, bias=False),
        )

    def forward(self, x):
        return self.lyapunov(x)


class Pvtol(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_model = PolicyNet()
        self.lyap_model = LyapunovNet()

    def forward(self, state):
        x = state[:, 0:1]
        y = state[:, 1:2]
        theta = state[:, 2:3]
        x_d = state[:, 3:4]
        y_d = state[:, 4:5]
        theta_d = state[:, 5:6]

        mass = env_params["mass"]
        inertia = env_params["inertia"]
        dist = env_params["dist"]
        gravity = env_params["gravity"]
        dt = env_params["dt"]
        max_force_ub_const = env_params["max_force_ub"]
        max_force_lb_const = env_params["max_force_lb"]

        action = self.policy_model(state)
        u = action + mass * gravity / 2.0
        u = max_force_lb_const + F.relu(u - max_force_lb_const)
        u = max_force_ub_const - F.relu(max_force_ub_const - u)
        u_1 = u[:, 0:1]
        u_2 = u[:, 1:2]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        x_change = x_d * cos_theta - y_d * sin_theta
        y_change = x_d * sin_theta + y_d * cos_theta
        theta_change = theta_d
        x_d_change = y_d * theta_d - gravity * sin_theta
        y_d_change = -x_d * theta_d - gravity * cos_theta + (u_1 + u_2) / mass
        theta_d_change = (u_1 - u_2) * dist / inertia

        x_next = x + dt * x_change
        y_next = y + dt * y_change
        theta_next = theta + dt * theta_change
        x_d_next = x_d + dt * x_d_change
        y_d_next = y_d + dt * y_d_change
        theta_d_next = theta_d + dt * theta_d_change
        state_next = torch.concat(
            [x_next, y_next, theta_next, x_d_next, y_d_next, theta_d_next], dim=-1
        )

        lyap = self.lyap_model(state)
        lyap_next = self.lyap_model(state_next)

        y_0 = lyap
        y_1 = lyap_next - lyap

        return torch.concat([y_0, y_1], dim=-1)
