import torch
import torch.nn as nn
import numpy as np
from scipy import linalg as la


class LinearController(nn.Module):
    """
    Simple linear controller.
    """

    def __init__(self, K, u_equilibrium, trainable=True, **kwargs):
        """
        Args:
          K: the coefficients of the linear controller.
          u_equilibrium: the controller output at equilibrium.
        """
        super().__init__()
        self.u_equilibrium = u_equilibrium
        self.trainable = trainable
        if trainable:
            self.register_parameter(
                name="K", param=torch.nn.Parameter(K.clone().detach())
            )
        else:
            self.K = K.clone().requires_grad_(False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.K, self.u_equilibrium)

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.u_equilibrium = fn(self.u_equilibrium)
        if not self.trainable:
            self.K = fn(self.K)
        return self


class NeuralNetworkController(nn.Module):
    def __init__(
        self,
        nlayer=3,
        in_dim=2,
        out_dim=1,
        hidden_dim=64,
        clip_output=None,
        u_lo=None,
        u_up=None,
        x_equilibrium=None,
        u_equilibrium=None,
        activation=nn.ReLU,
        *args,
        **kwargs
    ):
        """
        Simple neural network controller.

        The controller output is computed through following steps
        1. The neural network computes net(x)
        2. (a) If clip_output is "tanh", we truncate the network to within u_lo and
           u_up as f(x) = tanh(net(x)) * (u_up - u_lo)/2 + (u_up + u_lo)/2
           (b) If clip_output is "clamp", we truncate the network as
           f(x) = clamp(net(x), u_lo, u_up)
           (c) If clip_output is None, we set f(x) = net(x)
        3. If x_equilibrium and u_equilibrium are not None, we set
           u = f(x) - f(x*) + u*
           where x* is x_equilibrium, u* is u_equilibrium.
        This controller guarantees that the control action is within
        [u_lo, u_up] and at equilibrium state x_equilibrium, the control action
        is the equilibrium action u_equilibrium
        """
        super().__init__(*args, **kwargs)
        assert clip_output in (None, "tanh", "clamp")
        self.clip_output = clip_output
        if u_lo is not None:
            assert u_lo.shape == (out_dim,)
        self.u_lo = u_lo
        if u_up is not None:
            assert u_up.shape == (out_dim,)
        self.u_up = u_up
        if x_equilibrium is not None:
            assert x_equilibrium.shape == (in_dim,)
        self.x_equilibrium = x_equilibrium
        if u_equilibrium is not None:
            assert u_equilibrium.shape == (out_dim,)
            if self.u_lo is not None:
                assert torch.all(u_equilibrium >= self.u_lo)
            if self.u_up is not None:
                assert torch.all(u_equilibrium <= self.u_up)
        self.u_equilibrium = u_equilibrium
        layers = [nn.Linear(in_dim, out_dim if nlayer == 1 else hidden_dim)]
        for n in range(1, nlayer - 1):
            layers.append(activation())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        if nlayer != 1:
            layers.append(activation())
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.layers = layers
        # print(f'Controller function:')
        # print(self.net)

    def _unclipped_output(self, x: torch.Tensor) -> torch.Tensor:
        unclipped_output = self.net(x)
        if self.x_equilibrium is not None and self.u_equilibrium is not None:
            unclipped_output = (
                self.net(x) - self.net(self.x_equilibrium) + self.u_equilibrium
            )
        return unclipped_output

    def forward(self, x):
        unclipped_output = self._unclipped_output(x)

        if self.clip_output is None:
            return unclipped_output
        else:
            if self.clip_output == "tanh":
                # Apply tanh to make output between a certain bound.
                f = (
                    torch.tanh(self.net(x)) * (self.u_up - self.u_lo) / 2
                    + (self.u_lo + self.u_up) / 2
                )
                if self.x_equilibrium is not None and self.u_equilibrium is not None:
                    f_equilibrium = (
                        torch.tanh(self.net(self.x_equilibrium))
                        * (self.u_up - self.u_lo)
                        / 2
                        + (self.u_lo + self.u_up) / 2
                    )
                    return f - f_equilibrium + self.u_equilibrium
                else:
                    return f
            elif self.clip_output == "clamp":
                # Instead of calling clamp direct, we use relu twice. Currently auto_LIRPA doesn't handle clamp.
                f1 = torch.nn.functional.relu(unclipped_output - self.u_lo) + self.u_lo
                f = -(torch.nn.functional.relu(self.u_up - f1) - self.u_up)
                return f

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.x_equilibrium = fn(self.x_equilibrium)
        self.u_equilibrium = fn(self.u_equilibrium)
        if self.u_lo is not None:
            self.u_lo = fn(self.u_lo)
        if self.u_up is not None:
            self.u_up = fn(self.u_up)
        return self


class NeuralNetworkLuenbergerObserver(nn.Module):
    """
    Neural network observer that takes vectors as observations.
    The observer dynamics is z[t+1] = f(z[t], u[t]) + nn(z[t], y[t]-h(z[t]))
    """

    def __init__(
        self,
        z_dim,
        y_dim,
        dynamics,
        h,
        zero_obs_error,
        fc_hidden_dim=[16, 16, 8, 8],
        activation=nn.LeakyReLU,
    ):
        """
        z_dim: state estimate dimension, same as state dimension.
        y_dim: output dimension.
        h: observation function.
        zero_obs_error: y[t] - h(x[t]). when y[t] = h(z[t]), nn(z[t], y[t]-h(z[t])) = 0.
        """
        super().__init__()
        self.z_dim = z_dim
        self.dynamics = dynamics
        self.h = h
        self.zero_obs_error = zero_obs_error
        fc_layers = [nn.Linear(z_dim + y_dim, fc_hidden_dim[0])]
        n_fc_layers = len(fc_hidden_dim)
        for n in range(n_fc_layers - 1):
            fc_layers.append(activation())
            fc_layers.append(
                nn.Linear(
                    fc_hidden_dim[n],
                    fc_hidden_dim[n + 1] if n + 1 < n_fc_layers - 1 else z_dim,
                )
            )
        self.fc_net = nn.Sequential(*fc_layers)
        # print(f'Observer fully connected function:')
        # print(self.fc_net)

    def forward(self, z, u, y):
        batch_size = z.shape[0]
        K = torch.ones((batch_size, 1), device=self.zero_obs_error.device)
        z_nominal = self.dynamics(z, u)
        obs_error = y - self.h(z)
        Le = self.fc_net(torch.cat((z, obs_error), 1))
        L0 = self.fc_net(torch.cat((z, (K * self.zero_obs_error).to(z.device)), 1))
        unclipped_z = z_nominal + Le - L0
        return unclipped_z


class EKFObserver(nn.Module):
    def __init__(self, dynamics, h, gamma=0, delta=1e-3, lam=0, alpha=1.1):
        super().__init__()
        self.dynamics = dynamics
        self.nx = dynamics.continuous_time_system.nx
        self.nu = dynamics.nu
        self.ny = dynamics.continuous_time_system.ny
        self.h = h
        self.gamma = gamma
        self.delta = delta
        self.lam = lam
        self.Ix = torch.eye(self.nx)
        self.Iy = torch.eye(self.ny)
        self.alpha = alpha
        self.Q = self.delta * self.Ix
        self.R = self.delta * self.Iy
        self.P0, self.K0 = self.dare()

    def forward_varying_p(self, z0, Pk, u, y):
        batch_size = z0.shape[0]
        device = z0.device
        z1_prior = self.dynamics(z0, u)
        if not z1_prior.requires_grad:
            z1_prior.requires_grad = True
        Fk, _ = self.dynamics.linearized_dynamics(z0, u)

        h1_z = self.h(z1_prior)
        e1 = (y - h1_z).unsqueeze(-1)
        H1 = self.dynamics.continuous_time_system.linearized_observation(z1_prior)

        # Qk should actually depend on e0
        Qk = self.gamma * torch.kron(
            (e1.transpose(1, 2) @ e1), self.Ix.to(device)
        ) + self.delta * self.Ix.to(device)
        P1_prior = self.alpha**2 * Fk @ Pk @ Fk.transpose(1, 2) + Qk
        R1 = self.lam * H1 @ P1_prior @ H1.transpose(1, 2) + self.delta * self.Iy.to(
            device
        )
        K1 = (
            P1_prior
            @ H1.transpose(1, 2)
            @ torch.inverse(H1 @ P1_prior @ H1.transpose(1, 2) + R1)
        )
        z1 = z1_prior + (K1 @ e1).squeeze(-1)

        P1 = (self.Ix.to(device) - K1 @ H1) @ P1_prior @ (
            self.Ix.to(device) - K1 @ H1
        ).transpose(1, 2) + K1 @ R1 @ K1.transpose(1, 2)
        P_posterior = (self.Ix.to(device) - K1 @ H1) @ P1_prior
        return z1, P_posterior

    def forward(self, z0, u, y):
        batch_size = z0.shape[0]
        device = z0.device
        Pk = (torch.ones(batch_size, self.nx, self.nx) * self.P0).to(device)
        z1_prior = self.dynamics(z0, u)
        if not z1_prior.requires_grad:
            z1_prior.requires_grad = True
        Fk, _ = self.dynamics.linearized_dynamics(z0, u)

        h1_z = self.h(z1_prior)
        e1 = (y - h1_z).unsqueeze(-1)
        H1 = self.dynamics.continuous_time_system.linearized_observation(z1_prior)

        # Qk should actually depend on e0
        Qk = self.delta * self.Ix.to(device)
        P1_prior = self.alpha**2 * Fk @ Pk @ Fk.transpose(1, 2) + Qk
        R1 = self.lam * H1 @ P1_prior @ H1.transpose(1, 2) + self.delta * self.Iy.to(
            device
        )
        K1 = (
            P1_prior
            @ H1.transpose(1, 2)
            @ torch.linalg.inv(H1 @ P1_prior @ H1.transpose(1, 2) + R1)
        )
        z1 = z1_prior + (K1 @ e1).squeeze(-1)

        return z1

    def forward_constant_K(self, z0, u, y):
        K = (
            (torch.ones(z0.shape[0], self.K0.shape[0], self.K0.shape[1]) * self.K0)
            .to(y.device)
            .to(y.dtype)
        )
        z_prior = self.dynamics(z0, u)
        innovation = K @ (y - self.dynamics.continuous_time_system.h(z0)).unsqueeze(-1)
        z = z_prior + innovation.squeeze(-1)
        return z

    def dare(self):
        x0 = self.dynamics.x_equilibrium.unsqueeze(0)
        dtype = x0.dtype
        Ad, _ = self.dynamics.linearized_dynamics(
            x0, self.dynamics.u_equilibrium.unsqueeze(0)
        )
        Ad = Ad.squeeze().detach().numpy()
        Cd = (
            self.dynamics.continuous_time_system.linearized_observation(x0)
            .squeeze(0)
            .detach()
            .numpy()
        )
        Q = self.Q.cpu().detach().numpy()
        R = self.R.cpu().detach().numpy()
        P_prior = la.solve_discrete_are(Ad.T, Cd.T, Q, R)
        K = P_prior @ Cd.T @ np.linalg.inv(Cd @ P_prior @ Cd.T + R)
        return torch.tensor(P_prior, dtype=dtype), torch.tensor(K, dtype=dtype)
