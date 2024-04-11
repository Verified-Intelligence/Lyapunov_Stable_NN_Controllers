import math
import typing
from typing import Optional, Union

import torch.nn as nn
import torch
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system


def soft_max(x: torch.Tensor, beta: float = 100):
    x_max = torch.max(x, dim=-1, keepdim=True).values
    eq = torch.exp(beta * (x - x_max))
    if torch.any(torch.isnan(eq)):
        raise Exception("soft_max contains NAN, consider to reduce beta.")
    ret = torch.sum(eq / torch.sum(eq, dim=-1, keepdim=True) * x, dim=-1, keepdim=True)
    return ret


def logsumexp(x: torch.Tensor, beta: float = 100):
    x_max = torch.max(x, dim=-1, keepdim=True).values
    eq = torch.exp(beta * (x - x_max))
    if torch.any(torch.isnan(eq)):
        raise Exception("logsumexp contains NAN, consider to reduce beta.")
    return torch.log(torch.sum(eq, dim=-1, keepdim=True)) / beta


def soft_min(x: torch.Tensor, beta: float = 100):
    return -soft_max(-x, beta)


class NeuralNetworkLyapunov(nn.Module):
    """
    V(x) = V_nominal(x) + network_output(x) + V_psd_output(x)
    V_nominal(x) contains NO optimizable parameters.
    network_output =
    ϕ(x) − ϕ(x*) if absolute_output = False
    |ϕ(x) − ϕ(x*)| if absolute_output = True
    V_psd_output =
    |(εI+RᵀR)(x-x*)|₁ if V_psd_form = "L1"
    (x-x*)ᵀ(εI+RᵀR)(x-x*) if V_psd_form = "quadratic".
    |R(x-x*)|₁ if V_psd_form = "L1_R_free"

    The optimizable parameters are the network ϕ and R.
    """

    def __init__(
        self,
        goal_state: torch.Tensor,
        hidden_widths: list,
        x_dim: int,
        R_rows: int,
        absolute_output: bool,
        eps: float,
        activation: nn.Module,
        nominal: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        V_psd_form: str = "L1",
        *args,
        **kwargs
    ):
        """
        Args:
          hidden_widths: hidden_widths[i] is the width of the i'th hidden
          layer. This doesn't include the output layer, which always have
          width 1.
          x_dim: The dimension of state
          R_rows: The number of rows in matrix R.
          absolute_output: If absolute_output=False,
          then V(x) = V_nominal(x) + ϕ(x) − ϕ(x*) + |(εI+RᵀR)(x-x*)|₁
          otherwise V(x) = V_nominal(x) + |ϕ(x) − ϕ(x*)| + |(εI+RᵀR)(x-x*)|₁
          nominal: V_nominal(x) in the documentation above. If nominal=None,
          then we ignore V_nominal(x). Note that V_nominal(x*) should be 0.
          nominal(x) should support batch computation.
        """
        super().__init__(*args, **kwargs)
        self.goal_state = goal_state
        self.x_dim = x_dim
        assert self.goal_state.shape == (self.x_dim,)
        if hidden_widths is None:
            layers = []
        else:
            layers = [
                nn.Linear(
                    in_features=self.x_dim,
                    out_features=1 if len(hidden_widths) == 0 else hidden_widths[0],
                )
            ]
            for layer, width in enumerate(hidden_widths):
                layers.append(activation())
                layers.append(
                    nn.Linear(
                        in_features=width,
                        out_features=hidden_widths[layer + 1]
                        if layer != len(hidden_widths) - 1
                        else 1,
                    )
                )
            for l in layers:
                if isinstance(l, nn.Linear):
                    torch.nn.init.kaiming_uniform_(l.weight, nonlinearity="relu")
                    # print(f'layer max={l.weight.max().item()}, min={l.weight.min().item()}')
                    # l.weight.data.clamp_(min=-0.5, max=0.5)
        self.net = nn.Sequential(*layers)
        self.layers = layers
        assert isinstance(absolute_output, bool)
        self.absolute_output = absolute_output
        assert isinstance(eps, float)
        self.R_rows = R_rows
        # If R_rows is set to 0 we will not use R.
        if R_rows > 0:
            # assert (eps > 0)
            self.eps = eps
            # Rt is the transpose of R
            self.register_parameter(
                name="R",
                param=torch.nn.Parameter(torch.rand((R_rows, self.x_dim)) - 0.5),
            )
        self.nominal = nominal
        if self.nominal is not None:
            assert self.nominal(self.goal_state.unsqueeze(0))[0].item() == 0
        self.V_psd_form = V_psd_form

    def _network_output(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.net) > 0:
            phi = self.net(x)
            phi_star = self.net(self.goal_state)
            return phi - phi_star
        else:
            return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

    def _V_psd_output(self, x: torch.Tensor):
        """
        Compute
        |(εI+RᵀR)(x-x*)|₁
        or
        (x-x*)ᵀ(εI+RᵀR)(x-x*)
        or
        |R(x-x*)|₁
        """
        if self.R_rows > 0:
            eps_plus_RtR = self.eps * torch.eye(self.x_dim, device=x.device) + (
                self.R.transpose(0, 1) @ self.R
            )
            if self.V_psd_form == "L1":
                Rx = (x - self.goal_state) @ eps_plus_RtR
                # Use relu(x) + relu(-x) instead of torch.abs(x) since the verification code does relu splitting.
                l1_term = (
                    torch.nn.functional.relu(Rx) + torch.nn.functional.relu(-Rx)
                ).sum(dim=-1, keepdim=True)
                return l1_term
            elif self.V_psd_form == "quadratic":
                return torch.sum(
                    (x - self.goal_state) * ((x - self.goal_state) @ eps_plus_RtR),
                    dim=-1,
                    keepdim=True,
                )
            elif self.V_psd_form == "L1_R_free":
                Rx = (x - self.goal_state) @ self.R.transpose(0, 1)
                # Use relu(x) + relu(-x) instead of torch.abs(x) since the verification code does relu splitting.
                l1_term = (
                    torch.nn.functional.relu(Rx) + torch.nn.functional.relu(-Rx)
                ).sum(dim=-1, keepdim=True)
                return l1_term
            else:
                raise NotImplementedError
        else:
            return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    def forward(self, x):
        V_nominal = 0 if self.nominal is None else self.nominal(x)

        network_output = self._network_output(x)
        V_psd_output = self._V_psd_output(x)
        if self.absolute_output:
            return (
                V_nominal
                + torch.nn.functional.relu(network_output)
                + torch.nn.functional.relu(-network_output)
                + V_psd_output
            )
        else:
            return V_nominal + network_output + V_psd_output

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.goal_state = fn(self.goal_state)
        return self


class NeuralNetworkQuadraticLyapunov(nn.Module):
    """
    A quadratic Lyapunov function.
    This neural network output is
    V(x) = (x-x*)^T(εI+RᵀR)(x-x*),
    R is the parameters to be optimized.
    """

    def __init__(
        self,
        goal_state: torch.Tensor,
        x_dim: int,
        R_rows: int,
        eps: float,
        R: typing.Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        Args:
          x_dim: The dimension of state
          R_rows: The number of rows in matrix R.
          V(x) = (x-x*)^T(εI+RᵀR)(x-x*)
        """
        super().__init__(*args, **kwargs)
        self.goal_state = goal_state
        self.x_dim = x_dim
        assert self.goal_state.shape == (self.x_dim,)
        assert isinstance(eps, float)
        self.R_rows = R_rows
        assert eps >= 0
        self.eps = eps
        # Rt is the transpose of R
        if R is None:
            R = torch.rand((R_rows, self.x_dim)) - 0.5

        self.register_parameter(name="R", param=torch.nn.Parameter(R))

    def forward(self, x):
        x0 = x - self.goal_state
        Q = self.eps * torch.eye(self.x_dim, device=x.device) + (
            self.R.transpose(0, 1) @ self.R
        )
        return torch.sum(x0 * (x0 @ Q), axis=1, keepdim=True)

    def dVdx(self, x):
        Q = self.eps * torch.eye(self.x_dim, device=x.device) + (
            self.R.transpose(0, 1) @ self.R
        )
        dVdx = 2 * x @ Q
        return dVdx

    def diff(self, x, x_next, kappa, lyapunov_x):
        # V(x) = (x_t - x_*)^T Q (x_t - x_*)
        # V(x_next) = (x_next - x_*)^T Q (x_next - x_*)
        # dV = (x_next - x_*)^T Q (x_next - x_*) - (1-kappa) (x_t - x_*)^T Q (x_t - x_*)
        #    = x_next^T Q x_next
        #        - (1-kappa) x_t^T Q x_t
        #        - 2 (x_next - (1-kappa) x_t)^T Q x_*
        #        + kappa * x_*^T Q x_*
        #    = (x_next - sqrt(1-kappa) x_t)^T Q (x_next + sqrt(1-kappa) x_t)
        #        - 2 (x_next - (1-kappa)x_t)^T Q x_*
        #        + kappa * x_*^T Q x_*
        sqrt_1_minus_kappa = math.sqrt(1 - kappa)
        x_d1 = x_next - sqrt_1_minus_kappa * x
        if kappa == 0:
            x_d2 = x_d1
        else:
            x_d2 = x_next - (1 - kappa) * x
        x_s = x_next + sqrt_1_minus_kappa * x
        Q = (
            self.eps * torch.eye(self.x_dim, device=x.device)
            + (self.R.transpose(0, 1) @ self.R)
        )
        dV = (
            torch.sum(x_d1 * (x_s @ Q), axis=-1, keepdim=True)
            - 2 * torch.sum(x_d2 * (self.goal_state @ Q), axis=-1, keepdim=True)
            + kappa * torch.sum(self.goal_state * (self.goal_state @ Q), axis=-1, keepdim=True)
        )
        return dV

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.goal_state = fn(self.goal_state)
        return self


class LyapunovPositivityLoss(nn.Module):
    """
    Compute the loss V(x) - |N(x−x*)|₁
    where N is a given matrix.
    """

    def __init__(
        self, lyapunov: NeuralNetworkLyapunov, Nt: torch.Tensor, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lyapunov = lyapunov
        assert isinstance(Nt, torch.Tensor)
        assert Nt.shape[0] == self.lyapunov.x_dim
        self.Nt = Nt

    def forward(self, x):
        Nx = (x - self.lyapunov.goal_state) @ self.Nt
        # l1_term = (torch.nn.functional.relu(Nx) + torch.nn.functional.relu(-Nx)).sum(dim=1, keepdim=True)
        l1_term = torch.abs((x - self.lyapunov.goal_state) @ self.Nt).sum(
            dim=-1, keepdim=True
        )
        V = self.lyapunov(x)
        return V - l1_term

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.Nt = fn(self.Nt)


class LyapunovDerivativeSimpleLoss(nn.Module):
    """
    Require the Lyapunov function to always decrease, namely
    V(x_next) - V(x) <= -κ * V(x).

    We want to minimize
    V(x_next) - (1-κ)V(x)                (1)

    Since alpha-beta-crown verifies the quantity being non-negative rather than minimizing the loss, we compute the negation of (1) as
    (1-κ)V(x) - V(x_next)
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkLyapunov,
        kappa: float = 0.1,
        fuse_dV: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.fuse_dV = fuse_dV

    def forward(self, x, save_new_x=False):
        # Run the system by one step with dt.
        u = self.controller(x)
        new_x = self.dynamics.forward(x, u)
        if save_new_x:
            self.new_x = new_x
        lyapunov_x = self.lyapunov(x)
        self.last_lyapunov_x = lyapunov_x.detach()

        # The following two formulations are equivalent.
        # When self.fuse_dV is enabled, we fuse x and x_next (new_x)
        # before entering the quadratic term in the Lyapunov function to compute
        # tighter bounds on dV, compared to computing two Lyapunov function
        # values separately.
        if self.fuse_dV:
            assert isinstance(self.lyapunov, NeuralNetworkQuadraticLyapunov)
            dV = self.lyapunov.diff(x, new_x, self.kappa, lyapunov_x)
            loss = -dV
        else:
            loss = (1 - self.kappa) * lyapunov_x - self.lyapunov(new_x)

        return loss


class LyapunovDerivativeLoss(nn.Module):
    """
    For a box region B = [x_lo, x_up], we want enforce the condition that the
    set S={x in B | V(x)<=rho} is an invariant set, and V
    decreases within S.
    Namely we want the following conditions to hold:
    1. V(x)<= rho => x_next in B.
    2. V(x)<= rho => V(x_next) - V(x) <= -κ * V(x).
    This is equivalent to the following conditions:
    ((x_next in B) ∧ (V(x_next) - V(x) <= -κ * V(x))) ∨ (V(x) > rho)
    where "∨" is "logical or", and "∧" is "logical and".

    We can hence impose the loss
    min (weight[0]*(rho - V(x)),
         weight[1] * ReLU(V(x_next) - (1-κ)V(x))+
             weight[2] * (∑ᵢReLU((x_lo[i] - x_next[i]) + ReLU(x_next[i] - x_up[i]))))

    Since alpha-beta-crown wants to verify the loss being non-negative instead
    of minimizing the loss, we return the negation of the loss
    -min (weight[0] * (rho - V(x)),
         weight[1] * ReLU(V(x_next) - (1-κ)V(x)) +
             weight[2] * (∑ᵢReLU((x_lo[i] - x_next[i]) + ReLU(x_next[i] - x_up[i]))))
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkLyapunov,
        box_lo: torch.Tensor,
        box_up: torch.Tensor,
        rho_multiplier: float,
        kappa: float = 0.1,
        beta: float = 100,
        hard_max: bool = True,
        loss_weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            rho_multiplier: We use rho = rho_multiplier * min V(x_boundary)
            beta: the coefficient in soft max exponential. beta -> infinity
              recovers hard max.
            loss_weights: weight in the documentation above. Should all be non-negative.
        """
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyap_nn
        self.rho_multiplier = rho_multiplier
        self.box_lo = box_lo
        self.box_up = box_up
        self.kappa = kappa
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max
        if loss_weights is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            assert loss_weights.shape == (3,)
            assert torch.all(loss_weights > 0)
            self.loss_weights = loss_weights

    def get_rho(self):
        rho_boundary = self.lyapunov(self.x_boundary).min()
        rho = self.rho_multiplier * rho_boundary
        return rho

    def forward(self, x: torch.Tensor, save_new_x: bool = False):
        # Run the system by one step with dt.
        u = self.controller(x)
        new_x = self.dynamics.forward(x, u)
        if save_new_x:
            self.new_x = new_x
        lyapunov_x = self.lyapunov(x)
        rho = self.get_rho()
        loss1 = self.loss_weights[0] * (rho - lyapunov_x)
        loss2 = torch.nn.functional.relu(
            self.loss_weights[1]
            * (self.lyapunov(new_x) - (1 - self.kappa) * lyapunov_x)
        )
        loss3 = self.loss_weights[2] * (
            torch.nn.functional.relu(self.box_lo - new_x).sum(dim=1, keepdim=True)
            + torch.nn.functional.relu(new_x - self.box_up).sum(dim=1, keepdim=True)
        )
        loss23 = loss2 + loss3
        if self.hard_max:
            loss = torch.min(
                torch.cat((loss1, loss23), dim=-1),
                dim=-1,
                keepdim=True,
            ).values
            return -loss
        else:
            loss = soft_min(torch.cat((loss1, loss23), dim=-1), self.beta)
            return -loss


class LyapunovDerivativeSimpleLossWithV(LyapunovDerivativeSimpleLoss):
    """
    The same as LyapunovDerivativeSimpleLoss, but with V(x) as the second output.
    Used for verification with level set.
    TODO: make it a template class to also support LyapunovDerivativeDOFLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not hasattr(self, "x_boundary") or self.x_boundary is None

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        # Output should be [N, 2] where N is the batch size.
        return torch.cat((loss, self.last_lyapunov_x), dim=1)


class LyapunovDerivativeSimpleLossWithVBox(LyapunovDerivativeSimpleLossWithV):
    """
    Additionally, output x_next for checking if x_next is within the bounding box.
    """

    def forward(self, *args, **kwargs):
        loss_and_V = super().forward(*args, save_new_x=True, **kwargs)
        return torch.cat((loss_and_V, self.new_x), dim=1)


class LyapunovDerivativeDOFLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute (1-κ)*V(x, e) - V(x_next, e_next)
    V(x, e), e = x - z
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        observer,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkLyapunov,
        box_lo: torch.Tensor,
        box_up: torch.Tensor,
        rho_multiplier: float,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        loss_weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.observer = observer
        self.controller = controller
        self.lyapunov = lyap_nn
        self.rho_multiplier = rho_multiplier
        self.box_lo = box_lo
        self.box_up = box_up
        self.kappa = kappa
        self.nx = dynamics.continuous_time_system.nx
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max
        if loss_weights is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            assert loss_weights.shape == (3,)
            assert torch.all(loss_weights > 0)
            self.loss_weights = loss_weights

    def get_rho(self):
        rho_boundary = self.lyapunov(self.x_boundary).min()
        rho = self.rho_multiplier * rho_boundary
        return rho

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, : self.nx]
        e = xe[:, self.nx :]
        z = x - e
        y = self.observer.h(x)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        # u = self.controller(z)
        new_x = self.dynamics.forward(x, u)
        new_z = self.observer.forward(z, u, y)
        new_xe = torch.cat((new_x, new_x - new_z), dim=1)
        self.new_xe = new_xe
        lyapunov_x = self.lyapunov(xe)
        rho = self.get_rho()
        # Save the results for reference.
        self.last_lyapunov_x = lyapunov_x.detach()
        loss1 = self.loss_weights[0] * (rho - lyapunov_x)
        loss2 = self.loss_weights[1] * torch.nn.functional.relu(
            self.lyapunov(new_xe) - (1 - self.kappa) * lyapunov_x
        )
        loss3 = self.loss_weights[2] * (
            torch.nn.functional.relu(self.box_lo - new_xe).sum(dim=1, keepdim=True)
            + torch.nn.functional.relu(new_xe - self.box_up).sum(dim=1, keepdim=True)
        )
        loss23 = loss2 + loss3
        if self.hard_max:
            loss = torch.min(
                torch.cat((loss1, loss23), dim=-1),
                dim=-1,
                keepdim=True,
            ).values
            # if loss.sum() > 0:
            #     print(xe[(loss > 0).squeeze()]/self.box_up)
            #     print(loss2.sum(), loss3.sum())
            return -loss
        else:
            loss = soft_min(torch.cat((loss1, loss23), dim=-1), self.beta)
            return -loss

class LyapunovDerivativeDOFSimpleLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute (1-κ)*V(x, e) - V(x_next, e_next)
    V(x, e), e = x - z
    """

    def __init__(self,
                 dynamics: dynamical_system.DiscreteTimeSystem,
                 observer,
                 controller: controllers.NeuralNetworkController,
                 lyap_nn: NeuralNetworkLyapunov,
                 kappa=0.1,
                 beta: float = 100,
                 hard_max: bool = True,
                 fuse_dV: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.observer = observer
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.nx = dynamics.continuous_time_system.nx
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max
        self.fuse_dV = fuse_dV

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, :self.nx]
        e = xe[:, self.nx:]
        z = x - e
        y = self.observer.h(x)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        # u = self.controller(z)
        new_x = self.dynamics.forward(x, u)
        new_z = self.observer.forward(z, u, y)
        lyapunov_x = self.lyapunov(xe)
        # Save the results for reference.
        self.last_lyapunov_x = lyapunov_x.detach()
        self.new_xe = torch.cat((new_x, new_x - new_z), dim=1)

        # The following two formulations are equivalent.
        # When self.fuse_dV is enabled, we fuse x (xe) and x_next (self.new_xe)
        # before entering the quadratic term in the Lyapunov function to compute
        # tighter bounds on dV, compared to computing two Lyapunov function
        # values separately.
        if self.fuse_dV:
            assert isinstance(self.lyapunov, NeuralNetworkQuadraticLyapunov)
            dV = self.lyapunov.diff(xe, self.new_xe, self.kappa, lyapunov_x)
            loss = -dV
        else:
            loss = (1 - self.kappa) * lyapunov_x - self.lyapunov(self.new_xe)

        return loss


class LyapunovDerivativeDOFLossWithV(LyapunovDerivativeDOFSimpleLoss):
    """
    The same as LyapunovDerivativeLoss, but with V(x) as the second output.
    Used for verification with level set.
    TODO: make it a template class to also support LyapunovDerivativeDOFLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.x_boundary is None

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        # Output should be [N, 2] where N is the batch size.
        return torch.cat((loss, self.last_lyapunov_x), dim=1)


class LyapunovDerivativeDOFLossWithVBox(LyapunovDerivativeDOFLossWithV):
    """
    Additionally, output x_next for checking if x_next is within the bounding box.
    """

    def forward(self, *args, **kwargs):
        loss_and_v = super().forward(*args, **kwargs)
        return torch.cat((loss_and_v, self.new_xe), dim=1)


class LyapunovContinuousTimeDerivativeLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute -κ*V - ∂V/∂ ẋ >= 0
    """

    def __init__(
        self,
        continuous_time_system,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkQuadraticLyapunov,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.continuous_time_system = continuous_time_system
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.nx = continuous_time_system.nx
        self.nq = continuous_time_system.nq
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max

    def forward(self, x):
        # Run the system by one step with dt.
        u = self.controller.forward(x)
        v_dot = self.continuous_time_system.forward(x, u)
        x_dot = torch.cat((x[:, self.nq :], v_dot), dim=1)
        dVdx = self.lyapunov.dVdx(x)
        V_dot = torch.sum(dVdx * x_dot, dim=-1, keepdim=True)
        lyapunov_x = self.lyapunov(x)
        self.last_lyapunov_x = lyapunov_x.detach()
        loss = -self.kappa * lyapunov_x - V_dot
        if self.x_boundary is not None:
            q = torch.cat(
                (loss, lyapunov_x - self.lyapunov(self.x_boundary).min()), dim=-1
            )
            if self.hard_max:
                return torch.max(q, dim=-1, keepdim=True).values
            else:
                return soft_max(q, self.beta)

        else:
            return loss


class LyapunovContinuousTimeDerivativeLossWithV(LyapunovContinuousTimeDerivativeLoss):
    """
    The same as LyapunovDerivativeLoss, but with V(x) as the second output.
    Used for verification with level set.
    TODO: make it a template class to also support LyapunovDerivativeDOFLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.x_boundary is None

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        # Output should be [N, 2] where N is the batch size.
        return torch.cat((loss, self.last_lyapunov_x), dim=1)


class LyapunovContinuousTimeDerivativeDOFLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute -κ*V - ∂V/∂ ẋ >= 0
    """

    def __init__(
        self,
        continuous_time_system,
        observer,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkQuadraticLyapunov,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.continuous_time_system = continuous_time_system
        self.observer = observer
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.nx = continuous_time_system.nx
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, : self.nx]
        e = xe[:, self.nx :]
        z = x - e
        y = self.observer.h(x)
        # u = self.controller.forward(z)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        v_dot = self.continuous_time_system.forward(x, u)
        x_dot = torch.cat((x[:, self.nq :], v_dot), dim=1)
        z_dot = self.observer.forward(z, u, y)
        e_dot = x_dot - z_dot
        xi_dot = torch.cat((x_dot, e_dot), dim=1)
        dVdxi = self.lyapunov.dVdx(xe)
        V_dot = torch.sum(dVdxi * xi_dot, dim=-1, keepdim=True)
        lyapunov_x = self.lyapunov(xe)
        loss = -self.kappa * lyapunov_x - V_dot
        self.last_lyapunov_x = lyapunov_x.detach()
        if self.x_boundary is not None:
            q = torch.cat(
                (loss, lyapunov_x - self.lyapunov(self.x_boundary).min()), dim=-1
            )
            if self.hard_max:
                return torch.max(q, dim=-1, keepdim=True).values
            else:
                return soft_max(q, self.beta)

        else:
            return loss


class CLFDerivativeLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute -κ*V - ∂V/∂ ẋ >= 0
    """

    def __init__(
        self,
        continuous_time_system,
        lyap_nn,
        u_abs_box: torch.Tensor,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.continuous_time_system = continuous_time_system
        self.lyapunov = lyap_nn
        self.u_abs_box = u_abs_box
        self.kappa = kappa
        self.nx = continuous_time_system.nx
        self.nq = continuous_time_system.nq
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max

    def forward(self, x):
        # Run the system by one step with dt.
        f1 = self.continuous_time_system.f1(x)
        f2 = self.continuous_time_system.f2(x)
        dVdx = self.lyapunov.dVdx(x)
        Lf1 = torch.sum(dVdx * f1, dim=-1, keepdim=True)
        Lf2 = torch.sum(
            torch.abs(dVdx.unsqueeze(1) @ f2).squeeze() * self.u_abs_box,
            dim=-1,
            keepdim=True,
        )
        V_dot = Lf1 - Lf2
        lyapunov_x = self.lyapunov(x)
        self.last_lyapunov_x = lyapunov_x.detach()
        loss = -self.kappa * lyapunov_x - V_dot
        if self.x_boundary is not None:
            q = torch.cat(
                (loss, lyapunov_x - self.lyapunov(self.x_boundary).min()), dim=-1
            )
            if self.hard_max:
                return torch.max(q, dim=-1, keepdim=True).values
            else:
                return soft_max(q, self.beta)

        else:
            return loss


class ObserverLoss(nn.Module):
    """
    |x_next - x_ref_next|
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        observer,
        controller,
        ekf_observer,
        roll_out_steps=150,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.observer = observer
        self.controller = controller
        self.ekf_observer = ekf_observer
        self.nx = dynamics.continuous_time_system.nx
        self.roll_out_steps = roll_out_steps

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, : self.nx]
        e = xe[:, self.nx :]
        z = x - e
        y = self.observer.h(x)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        x_next = self.dynamics.forward(x, u)
        z_next = self.observer.forward(z, u, y)
        loss = torch.norm(x_next - z_next, p=2, dim=1)
        # z_ekf = self.ekf_observer.forward(z, u, y)
        # loss = torch.norm(z_ekf - z_next, p=2, dim=1)
        return loss.unsqueeze(1)


class LyapunovLowerBoundLoss:
    """
    We want the condition V(x) >= ρ
    We compute V(x) - ρ
    """

    def __init__(self, lyap: NeuralNetworkLyapunov, rho_roa: float):
        self.lyapunov = lyap
        self.rho_roa = rho_roa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lyapunov(x) - self.rho_roa
