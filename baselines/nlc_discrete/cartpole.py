import torch
import torch.nn as nn
import torch.nn.functional as F


env_params = {
    "gravity": 9.8,
    "masscart": 1.0,
    "masspole": 0.1,
    "total_mass": 1.1,
    "lengtorch": 1.0,
    "tau": 0.05,
    "max_force": 30.0,
}
STATE_SPACE_SIZE = 4


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.policy = nn.Linear(STATE_SPACE_SIZE, 1, bias=False)

    def forward(self, x):
        return self.policy(x)


class LyapunovNet(nn.Module):
    def __init__(self):
        super(LyapunovNet, self).__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False),
        )

    def forward(self, x):
        return self.lyapunov(x)


class Cartpole(nn.Module):
    def __init__(self, with_x_next=False):
        super().__init__()
        self.with_x_next = with_x_next
        self.policy_model = PolicyNet()
        self.lyap_model = LyapunovNet()

    def forward(self, state):
        action = self.policy_model(state)

        gravity = env_params["gravity"]
        masscart = env_params["masscart"]
        masspole = env_params["masspole"]
        total_mass = env_params["total_mass"]
        lengtorch = env_params["lengtorch"]
        tau = env_params["tau"]
        max_force = env_params["max_force"]

        force = action
        force = (-max_force) + F.relu(force - (-max_force))
        force = max_force - F.relu(max_force - force)

        x = state[:, 0:1]
        x_dot = state[:, 1:2]
        theta = state[:, 2:3]
        theta_dot = state[:, 3:4]

        costorcheta = torch.cos(theta)
        sintorcheta = torch.sin(theta)

        temp = masscart + masspole * sintorcheta**2
        thetaacc = (
            -force * costorcheta
            - masspole * lengtorch * theta_dot**2 * costorcheta * sintorcheta
            + total_mass * gravity * sintorcheta
        ) / (lengtorch * temp)
        xacc = (
            force
            + masspole
            * sintorcheta
            * (lengtorch * theta_dot**2 - gravity * costorcheta)
        ) / temp

        x_next = x + tau * x_dot
        x_dot_next = x_dot + tau * xacc
        theta_next = theta + tau * theta_dot
        theta_dot_next = theta_dot + tau * thetaacc

        state_next = torch.concat(
            [
                x_next,
                x_dot_next,
                theta_next,
                theta_dot_next,
            ],
            dim=-1,
        )

        lyap = self.lyap_model(state)
        lyap_next = self.lyap_model(state_next)

        y_0 = lyap
        y_1 = lyap_next - lyap
        out_of_hole = F.relu(torch.abs(state) - 0.1).sum(dim=-1, keepdim=True)

        if self.with_x_next:
            return torch.concat([y_0, y_1, out_of_hole, state_next], dim=-1)
        else:
            return torch.concat([y_0, y_1, out_of_hole], dim=-1)
