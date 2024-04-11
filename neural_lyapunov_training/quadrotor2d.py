import torch
import numpy as np
import scipy
import neural_lyapunov_training.models as models


class Quadrotor2DDynamics:
    """
    2D Quadrotor dynamics, based on https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor2d/quadrotor_2d.py
    """

    def __init__(
        self, length=0.25, mass=0.486, inertia=0.00383, gravity=9.81, *args, **kwargs
    ):
        self.nx = 6
        self.nu = 2
        # length of the rotor arm.
        self.length = length
        # mass of the quadrotor.
        self.mass = mass
        # moment of inertia
        self.inertia = inertia
        # gravity.
        self.gravity = gravity

    def forward(self, x, u):
        """
        Compute the continuous-time dynamics (batched, pytorch).
        This is the actual computation that will be bounded using auto_LiRPA.
        """
        q = x[:, :3]
        qdot = x[:, 3:]
        qddot1 = (-1.0 / self.mass) * (torch.sin(q[:, 2:]) * (u[:, :1] + u[:, 1:]))
        qddot2 = (1.0 / self.mass) * (
            torch.cos(q[:, 2:]) * (u[:, :1] + u[:, 1:])
        ) - self.gravity
        qddot3 = (self.length / self.inertia) * (u[:, :1] - u[:, 1:])
        return torch.cat((qddot1, qddot2, qddot3), dim=1)

    def f1(self, x):
        f1_tensor = torch.zeros(x.shape[0], self.nx, device=x.device)
        f1_tensor[:, :3] = x[:, 3:]
        f1_tensor[:, 4] = -self.mass * self.gravity
        return f1_tensor

    def f2(self, x):
        q = x[:, :3]
        f2_tensor = torch.zeros(x.shape[0], self.nx, self.nu, device=x.device)
        f2_tensor[:, 3, :] = (-1.0 / self.mass) * torch.sin(q[:, 2:])
        f2_tensor[:, 4, :] = (1.0 / self.mass) * torch.cos(q[:, 2:])
        f2_tensor[:, 5, 0] = self.length / self.inertia
        f2_tensor[:, 5, 1] = -self.length / self.inertia
        return f2_tensor

    def linearized_dynamics(self, x, u):
        """
        Return ∂ẋ/∂x and ∂ẋ/∂ u
        """
        if isinstance(x, np.ndarray):
            A = np.zeros((6, 6))
            B = np.zeros((6, 2))
            A[:3, 3:6] = np.eye(3)
            theta = x[2]
            A[3, 2] = -np.cos(theta) / self.mass * (u[0] + u[1])
            A[4, 2] = -np.sin(theta) / self.mass * (u[0] + u[1])
            B[3, 0] = -np.sin(theta) / self.mass
            B[3, 1] = B[3, 0]
            B[4, 0] = np.cos(theta) / self.mass
            B[4, 1] = B[4, 0]
            B[5, 0] = self.length / self.inertia
            B[5, 1] = -B[5, 0]
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            A = torch.zeros((6, 6), dtype=dtype)
            B = torch.zeros((6, 2), dtype=dtype)
            A[:3, 3:6] = torch.eye(3, dtype=dtype)
            theta = x[2]
            A[3, 2] = -torch.cos(theta) / self.mass * (u[0] + u[1])
            A[4, 2] = -torch.sin(theta) / self.mass * (u[0] + u[1])
            B[3, 0] = -torch.sin(theta) / self.mass
            B[3, 1] = B[3, 0]
            B[4, 0] = torch.cos(theta) / self.mass
            B[4, 1] = B[4, 0]
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

    @property
    def x_equilibrium(self):
        return torch.zeros((6,))

    @property
    def u_equilibrium(self):
        return torch.full((2,), (self.mass * self.gravity) / 2)


class Quadrotor2DLidarDynamics:
    """
    (y, theta, ydot, thetadot)
    2D Quadrotor dynamics, based on https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor2d/quadrotor_2d.py
    """

    def __init__(
        self,
        length=0.25,
        mass=0.486,
        inertia=0.00383,
        gravity=9.81,
        b=0,
        *args,
        **kwargs
    ):
        self.nx = 4
        self.nu = 2
        self.ny = 4
        # length of the rotor arm.
        self.length = length
        # mass of the quadrotor.
        self.mass = mass
        # moment of inertia
        self.inertia = inertia
        # gravity.
        self.gravity = gravity
        self.b = b
        self.H = 5
        self.angle_max = 0.149 * np.pi
        self.origin_height = 1

    def forward(self, x, u):
        """
        Compute the continuous-time dynamics (batched, pytorch).
        This is the actual computation that will be bounded using auto_LiRPA.
        """
        q = x[:, :2]
        qddot1 = (
            (1.0 / self.mass) * (torch.cos(q[:, 1:]) * (u[:, :1] + u[:, 1:]))
            - self.gravity
            - self.b * x[:, 2:3]
        )
        qddot2 = (self.length / self.inertia) * (u[:, :1] - u[:, 1:]) - self.b * x[
            :, 3:
        ]
        return torch.cat((qddot1, qddot2), dim=1)

    def linearized_dynamics(self, x, u):
        """
        Return ∂ẋ/∂x and ∂ẋ/∂ u
        """
        if isinstance(x, np.ndarray):
            A = np.zeros((4, 4))
            B = np.zeros((4, 2))
            A[:2, 2:4] = np.eye(2)
            theta = x[1]
            A[2, 1] = -np.sin(theta) / self.mass * (u[0] + u[1])
            A[2, 2] = -self.b
            A[3, 3] = -self.b
            B[2, 0] = np.cos(theta) / self.mass
            B[2, 1] = B[2, 0]
            B[3, 0] = self.length / self.inertia
            B[3, 1] = -B[3, 0]
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            device = x.device
            batch_size = x.shape[0]
            A = torch.zeros((batch_size, 4, 4), dtype=dtype)
            B = torch.zeros((batch_size, 4, 2), dtype=dtype)
            A[:, :2, 2:4] = torch.eye(2, dtype=dtype).repeat(batch_size, 1, 1)
            theta = x[:, 1]
            A[:, 2, 1] = -torch.sin(theta) / self.mass * (u[:, 0] + u[:, 1])
            A[:, 2, 2] = -self.b
            A[:, 3, 3] = -self.b
            B[:, 2, 0] = torch.cos(theta) / self.mass
            B[:, 2, 1] = B[:, 2, 0]
            B[:, 3, 0] = self.length / self.inertia
            B[:, 3, 1] = -B[:, 3, 0]
            return A.to(device), B.to(device)

    def lqr_control(self):
        """
        The control action should be u = K * (x - x*) + u*
        """
        x0 = self.x_equilibrium
        u0 = self.u_equilibrium
        Q = np.diag(np.array([1, 1, 0.5, 0.5]))
        R = np.diag(np.array([0.5, 0.5]))
        x_np = x0 if isinstance(x0, np.ndarray) else x0.detach().numpy()
        u_np = u0 if isinstance(u0, np.ndarray) else u0.detach().numpy()
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S

    def h(self, x):
        if isinstance(x, torch.Tensor):
            device = x.device
            batch_size = x.shape[0]
            y = (
                torch.ones(batch_size, self.ny, device=device) * x[:, :1]
                + self.origin_height
            )
            theta = x[:, 1:2]
            phi = theta - torch.linspace(
                -self.angle_max, self.angle_max, self.ny, device=device
            )
            # in_range = (phi >= -np.pi/2) * (phi <= np.pi/2)
            # lidar_rays = torch.ones(batch_size, self.ny, device=device) * self.H
            # lidar_rays[in_range] = y[in_range] / torch.cos(phi[in_range])
            lidar_rays = y / torch.cos(phi)
            lidar_rays = torch.nn.functional.relu(lidar_rays)
            lidar_rays = -torch.nn.functional.relu(self.H - lidar_rays) + self.H
        else:
            y = x[0]
            theta = x[1]
            phi = theta - np.linspace(-self.angle_max, self.angle_max, self.ny)
            in_range = (phi >= -np.pi / 2) and (phi <= np.pi / 2)
            lidar_rays = np.ones(self.ny) * self.H
            lidar_rays[in_range] = y / np.cos(phi[in_range])
            lidar_rays = np.clip(lidar_rays, 0, self.H)
        return lidar_rays

    def linearized_observation(self, x):
        device = x.device
        batch_size = x.shape[0]
        C = torch.zeros(batch_size, self.ny, self.nx, device=device)
        y = x[:, :1].repeat(1, self.ny) + self.origin_height
        theta = x[:, 1:2]
        phi = theta - torch.linspace(
            -self.angle_max, self.angle_max, self.ny, device=device
        )
        in_range = (phi >= -np.pi / 2) * (phi <= np.pi / 2)
        C[:, :, 0][in_range] = 1 / torch.cos(phi[in_range])
        C[:, :, 1][in_range] = (
            y[in_range] * torch.sin(phi[in_range]) / torch.cos(phi[in_range]) ** 2
        )
        lidar_rays = torch.ones(batch_size, self.ny, device=device)
        lidar_rays[in_range] = y[in_range] / torch.cos(phi[in_range])
        truncate = lidar_rays > self.H
        C[:, :][truncate] = 0
        return C

    def kalman_gain(self):
        from scipy import linalg as la

        Q = np.eye(self.nx) * 1e-3
        R = np.eye(self.ny) * 1e-3
        x0 = self.x_equilibrium.unsqueeze(0)
        A, _ = self.linearized_dynamics(x0, self.u_equilibrium.unsqueeze(0))
        A = A.squeeze().detach().numpy()
        C = self.linearized_observation(x0).squeeze().detach().numpy()
        P = la.solve_continuous_are(A.T, C.T, Q, R)
        L = P @ C.T @ np.linalg.inv(R)
        return L

    def lqg_cl(self):
        K, _ = self.lqr_control()
        A, B = self.linearized_dynamics(
            self.x_equilibrium.unsqueeze(0), self.u_equilibrium.unsqueeze(0)
        )
        A = A.squeeze().detach().numpy()
        B = B.squeeze().detach().numpy()
        L = self.kalman_gain()
        C = (
            self.linearized_observation(self.x_equilibrium.unsqueeze(0))
            .squeeze()
            .detach()
            .numpy()
        )
        Acl = np.vstack(
            (np.hstack((A + B @ K, -B @ K)), np.hstack((np.zeros([4, 4]), A - L @ C)))
        )
        Acl[np.abs(Acl) <= 1e-6] = 0
        return Acl

    @property
    def x_equilibrium(self):
        return torch.zeros((4,))

    @property
    def u_equilibrium(self):
        return torch.full((2,), (self.mass * self.gravity) / 2)


class Quadrotor2DVisualizer:
    """
    Copied from
    https://github.com/RussTedrake/underactuated/blob/master/underactuated/quadrotor2d.py
    """

    def __init__(self, ax, x_lim, y_lim):
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])

        self.length = 0.25  # moment arm (meters)

        self.base = np.vstack(
            (
                1.2 * self.length * np.array([1, -1, -1, 1, 1]),
                0.025 * np.array([1, 1, -1, -1, 1]),
            )
        )
        self.pin = np.vstack(
            (0.005 * np.array([1, 1, -1, -1, 1]), 0.1 * np.array([1, 0, 0, 1, 1]))
        )
        a = np.linspace(0, 2 * np.pi, 50)
        self.prop = np.vstack(
            (self.length / 1.5 * np.cos(a), 0.1 + 0.02 * np.sin(2 * a))
        )

        # yapf: disable
        self.base_fill = self.ax.fill(
            self.base[0, :], self.base[1, :], zorder=1, edgecolor="k",
            facecolor=[.6, .6, .6])
        self.left_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.right_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.left_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        self.right_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        # yapf: enable

    def draw(self, t, x):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])

        p = np.dot(R, self.base)
        self.base_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.base_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack((-self.length + self.pin[0, :], self.pin[1, :])))
        self.left_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]
        p = np.dot(R, np.vstack((self.length + self.pin[0, :], self.pin[1, :])))
        self.right_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack((-self.length + self.prop[0, :], self.prop[1, :])))
        self.left_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack((self.length + self.prop[0, :], self.prop[1, :])))
        self.right_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        self.ax.set_title("t = {:.1f}".format(t))
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("z (m)")
