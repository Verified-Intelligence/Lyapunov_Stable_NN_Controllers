import torch


class PathTrackingDynamics:
    def __init__(self, speed: float, length: float, radius: float):
        self.nx = 2
        self.nu = 1
        self.speed = speed
        self.length = length
        self.radius = radius

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        x: size is (batch, 2)
        u: size is (batch, 1)
        """

        theta_e = x[:, 1:2]
        sintheta_e = torch.sin(theta_e)
        costheta_e = torch.cos(theta_e)

        d_e_acc = self.speed * sintheta_e
        coef = self.radius / self.speed
        theta_e_acc = (self.speed * u / self.length) - (
            costheta_e / (coef - sintheta_e)
        )
        return torch.cat((d_e_acc, theta_e_acc), dim=1)

    def linearized_dynamics(self, x, u):
        device = x.device
        batch_size = x.shape[0]
        A = torch.zeros((batch_size, self.nx, self.nx), device=device)
        B = torch.zeros((batch_size, self.nx, self.nu), device=device)
        theta_e = x[:, 1:2]
        sintheta_e = torch.sin(theta_e)
        costheta_e = torch.cos(theta_e)
        coef = self.radius / self.speed
        A[:, 0, 1] = self.speed * costheta_e
        A[:, 1, 1] = -(sintheta_e * coef + 1) / ((coef - sintheta_e) ** 2)
        B[:, 1, 0] = self.speed / self.length
        return A.to(device), B.to(device)

    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.tensor([self.length / self.radius])
