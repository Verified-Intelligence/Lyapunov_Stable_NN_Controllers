import torch
import torch.nn as nn
from enum import Enum
import numpy as np
import control


class DiscreteTimeSystem(nn.Module):
    """
    Defines the interface for the discrete dynamical system.
    """

    def __init__(self, nx, nu, *args, **kwargs):
        super(DiscreteTimeSystem, self).__init__(*args, **kwargs)
        self.nx = nx
        self.nu = nu
        pass

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def x_equilibrium(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def u_equilibrium(self) -> torch.Tensor:
        raise NotImplementedError


class IntegrationMethod(Enum):
    ExplicitEuler = 1
    MidPoint = 2


class FirstOrderDiscreteTimeSystem(DiscreteTimeSystem):
    """
    This discrete-time system is constructed by discretizing a continuous time
    first-order dynamical system in time.
    """

    def __init__(
        self,
        continuous_time_system,
        dt: float,
        integration: IntegrationMethod = IntegrationMethod.ExplicitEuler,
    ):
        """
        Args:
          continuous_time_system: This system has to define a function
          xdot = f(x, u)
        """
        super(FirstOrderDiscreteTimeSystem, self).__init__(
            continuous_time_system.nx, continuous_time_system.nu
        )
        assert callable(getattr(continuous_time_system, "forward"))
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.dt = dt
        self.integration = integration
        self.continuous_time_system = continuous_time_system
        self.Ix = torch.eye(self.nx)

    def forward(self, x, u):
        """
        Compute x_next for a batch of x and u
        """
        assert x.shape[0] == u.shape[0]
        xdot = self.continuous_time_system.forward(x, u)
        if self.integration == IntegrationMethod.ExplicitEuler:
            x_next = x + xdot * self.dt
        else:
            raise NotImplementedError
        return x_next

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium


class SecondOrderDiscreteTimeSystem(DiscreteTimeSystem):
    """
    This discrete-time system is constructed by discretizing a continuous time
    second-order dynamical system in time.
    """

    def __init__(
        self,
        continuous_time_system,
        dt: float,
        position_integration: IntegrationMethod = IntegrationMethod.MidPoint,
        velocity_integration: IntegrationMethod = IntegrationMethod.ExplicitEuler,
    ):
        """
        Args:
          continuous_time_system: This system has to define a function
          qddot = f(x, u) where x = [q, qdot].
        """
        super(SecondOrderDiscreteTimeSystem, self).__init__(
            continuous_time_system.nx, continuous_time_system.nu
        )
        assert callable(getattr(continuous_time_system, "forward"))
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.nq = int(self.nx / 2)
        self.dt = dt
        self.velocity_integration = velocity_integration
        self.position_integration = position_integration
        self.continuous_time_system = continuous_time_system
        self.Ix = torch.eye(self.nx)

    def forward(self, x, u):
        """
        Compute x_next for a batch of x and u
        """
        assert x.shape[0] == u.shape[0]
        qddot = self.continuous_time_system.forward(x, u)
        if self.velocity_integration == IntegrationMethod.ExplicitEuler:
            qdot_next = x[:, self.nq :] + qddot * self.dt
        else:
            raise NotImplementedError
        if self.position_integration == IntegrationMethod.MidPoint:
            q_next = x[:, : self.nq] + (qdot_next + x[:, self.nq :]) / 2 * self.dt
        elif self.position_integration == IntegrationMethod.ExplicitEuler:
            q_next = x[:, : self.nq] + x[:, self.nq :] * self.dt
        else:
            raise NotImplementedError
        return torch.cat((q_next, qdot_next), dim=1)

    def linearized_dynamics(self, x, u):
        Ac, Bc = self.continuous_time_system.linearized_dynamics(x, u)
        Ad = self.dt * Ac + self.Ix.to(x.device)
        Bd = self.dt * Bc
        return Ad, Bd

    def output_feedback_linearized_lyapunov(self, K, L):
        """
        Given the control gain K and observer gain L, solve the discrete-time Lyapunov equation
        for the closed-loop system with the states and controls at equilibrium.
        The linearized dynamics are computed from the continuous-time system with Explicit Euler method.
        """
        x0 = self.x_equilibrium.unsqueeze(0)
        Ad, Bd = self.continuous_time_system.linearized_dynamics(
            x0, self.u_equilibrium.unsqueeze(0)
        )
        Ad = Ad.squeeze().detach().numpy()
        Bd = Bd.squeeze().detach().numpy()
        C = (
            self.continuous_time_system.linearized_observation(x0)
            .squeeze()
            .detach()
            .numpy()
        )
        Acl = np.vstack(
            (
                np.hstack((Ad + Bd @ K, -Bd @ K)),
                np.hstack((np.zeros([self.nx, self.nx]), Ad - L @ C)),
            )
        )
        Acl[np.abs(Acl) <= 1e-6] = 0
        S = control.dlyap(Acl, np.eye(2 * self.nx))
        return S

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium
