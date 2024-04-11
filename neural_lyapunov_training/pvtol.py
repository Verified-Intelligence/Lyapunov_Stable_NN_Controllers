import torch
import torch.nn as nn
import numpy as np
import scipy
import models


class PvtolDynamics(nn.Module):
    def __init__(
        self,
        length=0.25,
        mass=4.0,
        inertia=0.0475,
        gravity=9.8,
        dist=0.25,
        dt=0.05,
        *args,
        **kwargs
    ):
        super().__init__()
        self.nx = 6
        self.nq = 3
        self.nu = 2
        # length of the rotor arm.
        self.length = length
        # mass of the quadrotor.
        self.mass = mass
        # moment of inertia
        self.inertia = inertia
        # gravity.
        self.gravity = gravity
        self.dist = dist
        self.dt = dt

    def forward(self, state, u):
        """
        Compute the continuous-time dynamics (batched, pytorch).
        This is the actual computation that will be bounded using auto_LiRPA.
        """

        x = state[:, 0:1]
        y = state[:, 1:2]
        theta = state[:, 2:3]
        x_d = state[:, 3:4]
        y_d = state[:, 4:5]
        theta_d = state[:, 5:6]

        u_1 = u[:, 0:1]
        u_2 = u[:, 1:2]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        x_change = x_d * cos_theta - y_d * sin_theta
        y_change = x_d * sin_theta + y_d * cos_theta
        theta_change = theta_d
        x_d_change = y_d * theta_d - self.gravity * sin_theta
        y_d_change = -x_d * theta_d - self.gravity * cos_theta + (u_1 + u_2) / self.mass
        theta_d_change = (u_1 - u_2) * self.dist / self.inertia

        x_next = x + self.dt * x_change
        y_next = y + self.dt * y_change
        theta_next = theta + self.dt * theta_change
        x_d_next = x_d + self.dt * x_d_change
        y_d_next = y_d + self.dt * y_d_change
        theta_d_next = theta_d + self.dt * theta_d_change
        state_next = torch.concat(
            [x_next, y_next, theta_next, x_d_next, y_d_next, theta_d_next], dim=-1
        )

        return state_next

    def linearized_dynamics(self, x, u):
        # Appendix A.2 in "Neural Lyapunov Control for Discrete-Time Systems"
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        A[0, 3] = A[1, 4] = A[2, 5] = 1
        A[3, 2] = -self.gravity
        B[4, :] = 1.0 / self.mass
        B[5, 0] = self.length / self.inertia
        B[5, 1] = -B[5, 0]
        return A, B

    def lqr_control(self, Q, R, x, u):
        """
        The control action should be u = K * (x - x*) + u*
        """
        x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
        u_np = u if isinstance(u, np.ndarray) else u.detach().numpy()
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S

    def h(self, x):
        return x[:, :0]

    @property
    def x_equilibrium(self):
        return torch.zeros((6,))

    @property
    def u_equilibrium(self):
        return torch.full((2,), self.mass * self.gravity / 2)
