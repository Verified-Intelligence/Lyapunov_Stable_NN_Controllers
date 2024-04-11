import torch


class PendulumDynamics:
    """
    The inverted pendulum, with the upright equilibrium as the state origin.
    """

    def __init__(self, m: float = 1, l: float = 1, beta: float = 1, g: float = 9.81):
        self.nx = 2
        self.nu = 1
        self.ny = 1
        self.m = m  # Mass
        self.l = l  # Length
        self.g = g  # Gravity
        self.beta = beta  # Damping
        self.inertia = self.m * self.l**2

    def forward(self, x, u):
        """
        Dynamics. x: state (batch, 2); u: controller input (batch, 1).
        """
        # States (theta, thete_dot)
        theta, theta_dot = x[:, 0], x[:, 1]
        theta = theta.unsqueeze(-1)
        theta_dot = theta_dot.unsqueeze(-1)
        # Dynamics according to http://underactuated.mit.edu/pend.html
        ml2 = self.m * self.l * self.l
        d_theta = theta_dot
        d_theta_dot = (
            (-self.beta / ml2) * theta_dot
            + (self.g / self.l) * torch.sin(theta)
            + u / ml2
        )
        return d_theta_dot

    def linearized_dynamics(self, x, u):
        device = x.device
        batch_size = x.shape[0]
        A = torch.zeros((batch_size, self.nx, self.nx))
        B = torch.zeros((batch_size, self.nx, self.nu))
        A[:, 0, 1] = 1
        A[:, 1, 0] = self.g / self.l * torch.sin(x[:, 0])
        A[:, 1, 1] = -self.beta / (self.inertia)
        B[:, 1, 0] = 1 / self.inertia
        return A.to(device), B.to(device)

    def h(self, x):
        return x[:, : self.ny]

    def linearized_observation(self, x):
        batch_size = x.shape[0]
        C = torch.zeros(batch_size, self.ny, self.nx, device=x.device)
        C[:, 0] = 1
        return C

    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))
