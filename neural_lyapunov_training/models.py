# Definitions of dynamics (inverted pendulum, quadrotor2d), lyapunov function
# and loss function.
# Everything needs to be a subclass of nn.Module in order to be handled by
# auto_LiRPA.

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import torch
import scipy
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../torchdyn")

import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.quadrotor2d as quadrotor2d
import neural_lyapunov_training.path_tracking as path_tracking
import neural_lyapunov_training.pvtol as pvtol

from neural_lyapunov_training.controllers import LinearController


class Dynamics(nn.Module):
    """
    Base class for any dynamics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, u):
        raise NotImplementedError

    @property
    def x_equilibrium(self):
        raise NotImplementedError

    @property
    def u_equilibrium(self):
        raise NotImplementedError


class CartPoleDynamics(Dynamics):
    """
    The dynamics of a cart-pole with state x = [px, θ, px_dot, θdot]
    """

    def __init__(self, mc=10.0, mp=1.0, l=1.0, gravity=9.81, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc = mc
        self.mp = mp
        self.l = l
        self.gravity = gravity

    def forward(self, x, u):
        """
        Refer to https://underactuated.mit.edu/acrobot.html#cart_pole
        """
        px_dot = x[:, 2]
        theta_dot = x[:, 3]
        s = torch.sin(x[:, 1])
        c = torch.cos(x[:, 1])
        px_ddot = (
            u.squeeze(1) + self.mp * s * (self.l * theta_dot**2 + self.gravity * c)
        ) / (self.mp * s**2 + self.mc)
        theta_ddot = (
            -u.squeeze(1) * c
            - self.mp * self.l * theta_dot**2 * c * s
            - (self.mc + self.mp) * self.gravity * s
        ) / (self.l * self.mc + self.mp * s**2)
        return torch.cat(
            (
                px_dot.unsqueeze(1),
                theta_dot.unsqueeze(1),
                px_ddot.unsqueeze(1),
                theta_ddot.unsqueeze(1),
            ),
            dim=1,
        )


class AcrobotDynamics(Dynamics):
    """
    The dynamics of an Acrobot with state x = [θ1, θ2, θdot1, θdot2]
    """

    def __init__(
        self,
        m1=1.0,
        m2=1.0,
        l1=1.0,
        l2=2.0,
        lc1=0.5,
        lc2=1.0,
        Ic1=0.083,
        Ic2=0.33,
        b1=0.1,
        b2=0.1,
        gravity=9.81,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.Ic1 = Ic1
        self.Ic2 = Ic2
        self.b1 = b1
        self.b2 = b2
        self.gravity = gravity

        # We compute the minimal value of the mass matrix determinant. This is
        # useful later when we compute the division over mass
        # matrix determinant in the dynamics.
        # The determinant is Ic1*Ic2 + Ic1*m2*lc2*lc2 + Ic2*m1*lc1*lc1 +
        # Ic2*m2*l1*l1 + m1*m2*lc1*lc1*lc2*lc2
        # +m2*m2*lc2*lc2*l1*l1*sin(theta2)*sin(theta2)
        # We can safely ignore the last term (which is always non-negative) to
        # compute the minimal.
        self.det_M_minimal = (
            self.Ic1 * self.Ic2
            + self.Ic1 * self.m2 * self.lc2**2
            + self.Ic2 * self.m1 * self.lc1**2
            + self.Ic2 * self.m2 * self.l1**2
            + self.m1 * self.m2 * self.lc1**2 * self.lc2**2
        )

    def forward(self, x, u):
        """
        Compute the continuous-time dynamics.
        The dynamics is copied from
        https://github.com/RobotLocomotion/drake/blob/master/examples/acrobot/acrobot_plant.cc
        """
        assert x.shape[0] == u.shape[0]

        # The dynamics is M * vdot = B*u - bias
        c2 = torch.cos(x[:, 1])
        I1 = self.Ic1 + self.m1 * self.lc1 * self.lc1
        I2 = self.Ic2 + self.m2 * self.lc2 * self.lc2
        m2l1lc2 = self.m2 * self.l1 * self.lc2
        m12 = I2 + m2l1lc2 * c2

        M00 = I1 + I2 + self.m2 * self.l1 * self.l1 + 2 * m2l1lc2 * c2
        M01 = m12
        M10 = m12
        M11 = I2

        # C(q, v) * v terms.
        s1 = torch.sin(x[:, 0])
        s2 = torch.sin(x[:, 1])
        s12 = torch.sin(x[:, 0] + x[:, 1])
        bias0 = (
            -2 * m2l1lc2 * s2 * x[:, 3] * x[:, 2] + -m2l1lc2 * s2 * x[:, 3] * x[:, 3]
        )
        bias1 = m2l1lc2 * s2 * x[:, 2] * x[:, 2]
        # -gravity(q) terms.
        bias0 += self.gravity * self.m1 * self.lc1 * s1 + self.gravity * self.m2 * (
            self.l1 * s1 + self.lc2 * s12
        )
        bias1 += self.gravity * self.m2 * self.lc2 * s12
        # damping terms.
        bias0 += self.b1 * x[:, 2]
        bias1 += self.b2 * x[:, 3]

        # Compute rhs = B*u - bias
        rhs0 = -bias0
        rhs1 = u.squeeze(1) - bias1

        # Solve M * vdot = rhs as vdot = M_adj * rhs / det(M)
        # To hint the verifier that det_M is strictly positive, we use ReLU to
        # bound it from below.
        det_M = (
            torch.nn.functional.relu(M00 * M11 - M01 * M10 - self.det_M_minimal)
            + self.det_M_minimal
        )
        M_adj00 = M11
        M_adj01 = -M10
        M_adj10 = -M01
        M_adj11 = M00
        vdot0 = (M_adj00 * rhs0 + M_adj01 * rhs1) / det_M
        vdot1 = (M_adj10 * rhs0 + M_adj11 * rhs1) / det_M
        v = x[:, 2:]
        return torch.cat((v, vdot0.unsqueeze(-1), vdot1.unsqueeze(-1)), dim=1)

    @property
    def x_equilibrium(self):
        return torch.zeros((4,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))


class Quadrotor3DDynamics(Dynamics):
    """
    3D Quadrotor dyanamics, based on https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor3d/quadrotor.py
    Modified to support batch computation and auto_LiRPA bounding.
    """

    def __init__(self, dtype):
        """
        The parameter of this quadrotor is obtained from
        Attitude stabilization of a VTOL quadrotor aircraft
        by Abdelhamid Tayebi and Stephen McGilvray.
        """
        super().__init__()
        self.mass = 0.468
        self.gravity = 9.81
        self.arm_length = 0.225
        # The inertia matrix is diagonal, we only store Ixx, Iyy and Izz.
        self.inertia = torch.tensor([4.9e-3, 4.9e-3, 8.8e-3], dtype=dtype)
        # The ratio between the torque along the z axis versus the force.
        self.z_torque_to_force_factor = 1.1 / 29
        self.dtype = dtype
        self.hover_thrust = self.mass * self.gravity / 4

    def rpy2rotmat(self, rpy):
        rpy_0 = rpy[:, 0:1]
        rpy_1 = rpy[:, 1:2]
        rpy_2 = rpy[:, 2:3]
        cos_roll = torch.cos(rpy_0)
        sin_roll = torch.sin(rpy_0)
        cos_pitch = torch.cos(rpy_1)
        sin_pitch = torch.sin(rpy_1)
        cos_yaw = torch.cos(rpy_2)
        sin_yaw = torch.sin(rpy_2)

        # Return a 3x3 tuple.
        results = (
            (
                cos_pitch * cos_yaw,
                -cos_roll * sin_yaw + cos_yaw * sin_pitch * sin_roll,
                cos_roll * cos_yaw * sin_pitch + sin_roll * sin_yaw,
            ),
            (
                cos_pitch * sin_yaw,
                cos_roll * cos_yaw + sin_pitch * sin_roll * sin_yaw,
                cos_roll * sin_pitch * sin_yaw - cos_yaw * sin_roll,
            ),
            (-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll),
        )
        return results

    def cross(self, a, b):
        # 3-d cross-product of a and b. a and b must have shape [batch, 3].
        a1 = a[:, 0:1]
        a2 = a[:, 1:2]
        a3 = a[:, 2:3]
        b1 = b[:, 0:1]
        b2 = b[:, 1:2]
        b3 = b[:, 2:3]
        s1 = a2 * b3 - a3 * b2
        s2 = a3 * b1 - a1 * b3
        s3 = a1 * b2 - a2 * b1
        return torch.cat((s1, s2, s3), dim=-1)

    def forward(self, x, u):
        # Compute the time derivative of the state.
        # The dynamics is explained in
        # Minimum Snap Trajectory Generation and Control for Quadrotors
        # by Daniel Mellinger and Vijay Kumar
        # @param x current system state, in shape [batch, 12]
        #          the states are [pos_x, pos_y, pos_z, roll, pitch, yaw,
        #          pos_xdot, pos_ydot, pos_zdot, angular_vel_x,
        #          angular_vel_y, angular_vel_z]
        # @param u the thrust generated in each propeller, its shape is
        #          [batch, 4]
        rpy = x[:, 3:6]
        pos_dot = x[:, 6:9]
        omega = x[:, 9:12]

        # plant_input is [total_thrust, torque_x, torque_y, torque_z]
        total_thrust = u.sum(dim=-1, keepdim=True)
        torque_x = self.arm_length * u[:, 1:2] - self.arm_length * u[:, 3:4]
        torque_y = -self.arm_length * u[:, 0:1] + self.arm_length * u[:, 2:3]
        torque_z = self.z_torque_to_force_factor * (
            u[:, 0:1] - u[:, 1:2] + u[:, 2:3] - u[:, 3:4]
        )
        torque = torch.cat((torque_x, torque_y, torque_z), dim=-1)

        R = self.rpy2rotmat(rpy)
        # We actually only need the last column of R.
        R_col = R[0][2], R[1][2], R[2][2]
        pos_ddot_0 = R_col[0] * total_thrust / self.mass
        pos_ddot_1 = R_col[1] * total_thrust / self.mass
        pos_ddot_2 = R_col[2] * total_thrust / self.mass - self.gravity

        # Here we exploit the fact that the inertia matrix is diagonal.
        omega_dot = (self.cross(-omega, self.inertia * omega) + torque) / self.inertia
        # Convert the angular velocity to the roll-pitch-yaw time
        # derivative.
        rpy_0 = rpy[:, 0:1]
        rpy_1 = rpy[:, 1:2]
        cos_roll = torch.cos(rpy_0)
        sin_roll = torch.sin(rpy_0)
        cos_pitch = torch.cos(rpy_1)
        tan_pitch = torch.tan(rpy_1)

        # Equation 2.7 in quadrotor control: modeling, nonlinear control
        # design and simulation by Francesco Sabatino
        omega_0 = omega[:, 0:1]
        omega_1 = omega[:, 1:2]
        omega_2 = omega[:, 2:3]
        rpy_dot_0 = (
            omega_0 + sin_roll * tan_pitch * omega_1 + cos_roll * tan_pitch * omega_2
        )
        rpy_dot_1 = cos_roll * omega_1 - sin_roll * omega_2
        rpy_dot_2 = (sin_roll / cos_pitch) * omega_1 + (cos_roll / cos_pitch) * omega_2
        return torch.cat(
            (
                pos_dot,
                rpy_dot_0,
                rpy_dot_1,
                rpy_dot_2,
                pos_ddot_0,
                pos_ddot_1,
                pos_ddot_2,
                omega_dot,
            ),
            dim=-1,
        )

    @property
    def x_equilibrium(self):
        return torch.zeros((12,), dtype=self.dtype)

    @property
    def u_equilibrium(self):
        return torch.full((4,), self.hover_thrust, dtype=self.dtype)

    def linearized_dynamics(self, x: np.ndarray, u: np.ndarray):
        """
        Return ∂ẋ/∂x and ∂ẋ/∂u
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(u, np.ndarray)
        x_torch = torch.from_numpy(x).reshape((1, -1)).to(self.inertia.device)
        u_torch = torch.from_numpy(u).reshape((1, -1)).to(self.inertia.device)
        x_torch.requires_grad = True
        u_torch.requires_grad = True
        xdot = self.forward(x_torch, u_torch)
        A = np.empty((12, 12))
        B = np.empty((12, 4))
        for i in range(12):
            if x_torch.grad is not None:
                x_torch.grad.zero_()
            if u_torch.grad is not None:
                u_torch.grad.zero_()
            xdot[0, i].backward(retain_graph=True)
            A[i, :] = x_torch.grad[0, :].detach().numpy()
            B[i, :] = u_torch.grad[0, :].detach().numpy()
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

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.inertia = fn(self.inertia)
        return self


class QuadraticLyapunov(nn.Module):
    """
    Simple quadratic Lyapunov function.
    """

    def __init__(self, S, **kwargs):
        """
        Args:
          S: the matrix specifying quadratic function x^T S x as the Lyapunov function.
        """
        super().__init__()
        self.register_parameter(name="S", param=torch.nn.Parameter(S.clone().detach()))

    def forward(self, x):
        return torch.sum(x * (x @ self.S), axis=1, keepdim=True)


def create_model(
    dynamics,
    controller_parameters=None,
    lyapunov_parameters=None,
    loss_parameters=None,
    path=None,
    lyapunov_func="lyapunov.NeuralNetworkLyapunov",
    loss_func="lyapunov.LyapunovDerivativeLoss",
    controller_func="controllers.NeuralNetworkController",
):
    """
    Build the computational graph for verification of general dynamics + controller + neural lyapunov function.
    """
    # Default parameters.
    if controller_parameters is None:
        controller_parameters = {
            "nlayer": 2,
            "hidden_dim": 5,
            "clip_output": None,
        }
    if lyapunov_parameters is None:
        lyapunov_parameters = {
            # 'nlayer': 3,
            "hidden_widths": [32, 32],
            "R_rows": 0,
            "absolute_output": False,
            "eps": 0.0,
            "activation": nn.ReLU,
        }
    if loss_parameters is None:
        loss_parameters = {
            "kappa": 0.1,
        }
    controller = eval(controller_func)(
        in_dim=dynamics.x_equilibrium.size(0),
        out_dim=dynamics.u_equilibrium.size(0),
        x_equilibrium=dynamics.x_equilibrium,
        u_equilibrium=dynamics.u_equilibrium,
        **controller_parameters,
    )
    lyapunov_nn = eval(lyapunov_func)(
        x_dim=dynamics.x_equilibrium.size(0),
        goal_state=dynamics.x_equilibrium,
        **lyapunov_parameters,
    )

    loss = eval(loss_func)(dynamics, controller, lyapunov_nn, **loss_parameters)
    # TODO: load a trained model. Currently using random weights, just to test autoLiRPA works.
    if path is not None:
        loss.load_state_dict(torch.load(path))
    return loss


def create_output_feedback_model(
    dynamics,
    controller_parameters,
    lyapunov_parameters,
    path=None,
    observer_parameters=None,
    loss_parameters=None,
    lyapunov_func="lyapunov.NeuralNetworkLyapunov",
    loss_func="lyapunov.LyapunovDerivativeDOFLoss",
    controller_func="controllers.NeuralNetworkController",
    observer_func="controllers.NeuralNetworkLuenbergerObserver",
):
    """
    Build the computational graph for verification of general dynamics + controller + neural lyapunov function.
    """
    if loss_parameters is None:
        loss_parameters = {
            "kappa": 0,
        }
    nx = dynamics.continuous_time_system.nx
    ny = dynamics.continuous_time_system.ny
    nu = dynamics.continuous_time_system.nu
    h = lambda x: dynamics.continuous_time_system.h(x)
    controller = eval(controller_func)(
        in_dim=nx + ny,
        out_dim=nu,
        x_equilibrium=torch.concat((dynamics.x_equilibrium, torch.zeros(ny))),
        u_equilibrium=dynamics.u_equilibrium,
        **controller_parameters,
    )
    lyapunov_nn = eval(lyapunov_func)(
        x_dim=2 * nx,
        goal_state=torch.concat((dynamics.x_equilibrium, torch.zeros(nx))),
        **lyapunov_parameters,
    )
    observer = eval(observer_func)(
        nx, ny, dynamics, h, torch.zeros(1, ny), observer_parameters["fc_hidden_dim"]
    )
    loss = eval(loss_func)(
        dynamics, observer, controller, lyapunov_nn, **loss_parameters
    )
    if path is not None:
        loss.load_state_dict(torch.load(path)["state_dict"])
    return loss


def create_pendulum_model_state_feedback(**kwargs):
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1),
            dt=0.05,
            position_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
            velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
        ),
        **kwargs,
    )


def create_pendulum_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1), dt
        ),
        **kwargs,
    )


def create_pendulum_output_feedback_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_output_feedback_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1), dt
        ),
        **kwargs,
    )


def create_path_tracking_model(dt=0.05, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_model(
        dynamical_system.FirstOrderDiscreteTimeSystem(
            path_tracking.PathTrackingDynamics(speed=2.0, length=1.0, radius=10.0), dt
        ),
        **kwargs,
    )


def create_quadrotor2d_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the Quadrotor2D model.
    """
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            quadrotor2d.Quadrotor2DDynamics(),
            dt=dt,
        ),
        # pretrained_path='lyaloss_quadroter2d.pth',
        **kwargs,
    )


def create_quadrotor2d_output_feedback_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the Quadrotor2D model.
    """
    return create_output_feedback_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            quadrotor2d.Quadrotor2DLidarDynamics(
                length=0.25, mass=0.486, inertia=0.00383, gravity=9.81
            ),
            dt=dt,
        ),
        **kwargs,
    )


def create_continuous_time_quadrotor2d_model(**kwargs):
    """
    Build the computational graph for verification of the Quadrotor2D model.
    """
    return create_model(
        quadrotor2d.Quadrotor2DDynamics(
            length=0.25, mass=0.486, inertia=0.00383, gravity=9.81
        ),
        # pretrained_path='lyaloss_quadroter2d.pth',
        **kwargs,
    )


def create_pvtol_model(**kwargs):
    """
    Build the computational graph for verification of the Pvtol model.
    """
    return create_model(pvtol.PvtolDynamics(), **kwargs)


def create_quadrotor3d_model(**kwargs):
    """
    Build the computational graph for verification of the Quadrotor3D model.
    """
    return create_model(
        Quadrotor3DDynamics(dtype=torch.float32),
        # pretrained_path='lyaloss_quadroter2d.pth',
        **kwargs,
    )


def create_cartpole_model(**kwargs):
    """
    Build the computational graph for verification of the Acrobot model.
    """
    return create_model(
        CartPoleDynamics(),
        controller_parameters={"nlayer": 1},
        # pretrained_path='lyaloss_acrobot.pth',
        **kwargs,
    )


def create_acrobot_model(**kwargs):
    """
    Build the computational graph for verification of the Acrobot model.
    """
    return create_model(
        AcrobotDynamics(),
        controller_parameters={"nlayer": 1},
        # pretrained_path='lyaloss_acrobot.pth',
        **kwargs,
    )


def add_hole(box_low, box_high, inner_low, inner_high):
    boxes_low = []
    boxes_high = []
    for i in range(box_low.size(0)):
        # Split on dimension i.
        box1_low = box_low.clone()
        box1_low[i] = inner_high[i]
        box1_high = box_high.clone()
        box2_low = box_low.clone()
        box2_high = box_high.clone()
        box2_high[i] = inner_low[i]
        boxes_low.extend([box1_low, box2_low])
        boxes_high.extend([box1_high, box2_high])
        box_low[i] = inner_low[i]
        box_high[i] = inner_high[i]
    boxes_low = torch.stack(boxes_low, dim=0)
    boxes_high = torch.stack(boxes_high, dim=0)
    return boxes_low, boxes_high


def box_data(
    eps=None, lower_limit=-1.0, upper_limit=1.0, ndim=2, scale=1.0, hole_size=0
):
    """
    Generate a box between (-1, -1) and (1, 1) as our region to verify stability.
    We may place a small hole around the origin.
    """
    if isinstance(lower_limit, list):
        data_min = scale * torch.tensor(
            lower_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_min = scale * torch.ones((1, ndim)) * lower_limit
    if isinstance(upper_limit, list):
        data_max = scale * torch.tensor(
            upper_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_max = scale * torch.ones((1, ndim)) * upper_limit
    if hole_size != 0:
        inner_low = data_min.squeeze(0) * hole_size
        inner_high = data_max.squeeze(0) * hole_size
        data_min, data_max = add_hole(
            data_min.squeeze(0), data_max.squeeze(0), inner_low, inner_high
        )
    X = (data_min + data_max) / 2.0
    # Assume the "label" is 1, so we verify the positiveness.
    labels = torch.ones(size=(data_min.size(0),), dtype=torch.int64)
    # Lp norm perturbation epsilon. Not used, since we will return per-element min and max.
    eps = None
    return X, labels, data_max, data_min, eps


def simulate(lyaloss: lyapunov.LyapunovDerivativeLoss, steps: int, x0):
    # Assumes explicit euler integration.
    x_traj = [None] * steps
    V_traj = [None] * steps
    x_traj[0] = x0
    with torch.no_grad():
        V_traj[0] = lyaloss.lyapunov.forward(x_traj[0])
        for i in range(1, steps):
            u = lyaloss.controller.forward(x_traj[i - 1])
            x_traj[i] = lyaloss.dynamics.forward(x_traj[i - 1], u)
            V_traj[i] = lyaloss.lyapunov.forward(x_traj[i])

    return x_traj, V_traj
