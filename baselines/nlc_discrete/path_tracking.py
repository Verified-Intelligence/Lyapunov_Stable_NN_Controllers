import torch
import torch.nn as nn
import torch.nn.functional as F


env_params = {
    "speed": 2.0,
    "length": 1.0,
    "radius": 10.0,
    "dt": 0.05,
    "max_force": 0.84,
}
STATE_SPACE_SIZE = 2


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.policy(x)


class LyapunovNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.lyapunov(x)


class PathTracking(nn.Module):
    def __init__(self, with_x_next):
        super().__init__()
        self.with_x_next = with_x_next
        self.policy_model = PolicyNet()
        self.lyap_model = LyapunovNet()

    def forward(self, x):
        speed = env_params["speed"]
        radius = env_params["radius"]
        length = env_params["length"]
        max_force = env_params["max_force"]
        dt = env_params["dt"]
        coef = radius / speed

        theta = x[:, 1:2]
        sin_theta = torch.sin(theta)
        force_mid = self.policy_model(x)

        # TODO support clamp natively
        # force = force_mid.clamp(min=-max_force, max=max_force)
        force = -max_force + F.relu(force_mid - (-max_force))
        force = max_force - F.relu(max_force - force)

        d_e_acc = env_params["speed"] * sin_theta
        # TODO may be better to bound as a whole
        theta_e_acc = speed * force / length - torch.cos(theta) / (coef - sin_theta)

        x_next = x + torch.concat([d_e_acc, theta_e_acc], dim=-1) * dt

        lyp_val_x0 = self.lyap_model(x)
        lyp_val_x0_next = self.lyap_model(x_next)

        y_0 = lyp_val_x0
        y_1 = lyp_val_x0_next - lyp_val_x0
        out_of_hole = F.relu(torch.abs(x) - 0.1).sum(dim=-1, keepdim=True)

        if self.with_x_next:
            return torch.concat([y_0, y_1, out_of_hole, x_next], dim=-1)
        else:
            return torch.concat([y_0, y_1, out_of_hole], dim=-1)
