import torch
import torch.nn as nn
import torch.nn.functional as F


env_params = {
    "max_torque": 6.0,
    "dt": 0.05,
    "gravity": 9.81,
    "mass": 0.15,
    "length": 0.5,
}
STATE_SPACE_SIZE = 2


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(STATE_SPACE_SIZE, 1, bias=False)

    def forward(self, x):
        return self.policy(x)


class LyapunovNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, 8), nn.ReLU(), nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.lyapunov(x)


class Pendulum(nn.Module):
    def __init__(self, with_x_next=False):
        super().__init__()
        self.with_x_next = with_x_next
        self.policy_model = PolicyNet()
        self.lyap_model = LyapunovNet()

    def forward(self, x):
        max_torque = env_params["max_torque"]
        dt = env_params["dt"]
        gravity = env_params["gravity"]
        mass = env_params["mass"]
        length = env_params["length"]

        force = self.policy_model(x)
        force = -max_torque + F.relu(force - (-max_torque))
        force = max_torque - F.relu(max_torque - force)
        theta = x[:, 0:1]
        theta_dot = x[:, 1:2]
        sintheta = torch.sin(theta)
        thetaacc = (mass * gravity * length * sintheta + force - 0.1 * theta_dot) / (
            mass * length * length
        )
        theta_next = theta + dt * theta_dot
        theta_dot_next = theta_dot + dt * thetaacc
        x_next = torch.cat((theta_next, theta_dot_next), dim=1)

        lyp_val_x0 = self.lyap_model(x)
        lyp_val_x0_next = self.lyap_model(x_next)

        y_0 = lyp_val_x0
        y_1 = lyp_val_x0_next - lyp_val_x0
        out_of_hole = F.relu(torch.abs(x) - 0.1).sum(dim=-1, keepdim=True)

        if self.with_x_next:
            return torch.concat([y_0, y_1, out_of_hole, x_next], dim=-1)
        else:
            return torch.concat([y_0, y_1, out_of_hole], dim=-1)
