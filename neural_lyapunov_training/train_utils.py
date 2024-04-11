from dataclasses import dataclass
import itertools
import typing
from typing import Optional
import random
import time
from numpy.random import wald
import logging

import torch
import numpy as np
import wandb
from auto_LiRPA import BoundedTensor, BoundedModule
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.eps_scheduler import SmoothedScheduler
import matplotlib.pyplot as plt
from matplotlib import cm
import neural_lyapunov_training.lyapunov as lyapunov


def generate_grids(lower_limit, upper_limit, grid_size):
    ndim = lower_limit.size(0)
    assert ndim == upper_limit.size(0)
    assert lower_limit.ndim == upper_limit.ndim == 1
    grids = [None] * ndim
    steps = (upper_limit - lower_limit) / grid_size
    for d in range(ndim):
        grids[d] = torch.linspace(
            lower_limit[d], upper_limit[d], grid_size[d] + 1, device=lower_limit.device
        )[: grid_size[d]]
    lower = torch.cartesian_prod(*grids)
    upper = lower + steps
    return lower, upper


def generate_grids_on_box_boundary(
    lower_boundary: torch.Tensor, upper_boundary: torch.Tensor, grid_size: torch.Tensor
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate grids on the box boundary.

    Return: (lower, upper) The i'th row is the lower and upper bound of the
    i'th grid cell.
    """
    x_dim = lower_boundary.numel()
    lower = torch.empty((0, x_dim))
    upper = torch.empty((0, x_dim))
    # Loop through each face
    for i in range(x_dim):
        grid_size_i = grid_size.clone()
        grid_size_i[i] = 1
        # First generate grid on the i'th coordinate lower plane.
        upper_boundary_i = upper_boundary.clone()
        upper_boundary_i[i] = lower_boundary[i]
        lower_i, upper_i = generate_grids(lower_boundary, upper_boundary_i, grid_size_i)
        lower = torch.cat((lower, lower_i), dim=0)
        upper = torch.cat((upper, upper_i), dim=0)
        # Now generate grid on the i'th coordinate upper plane
        lower_boundary_i = lower_boundary.clone()
        lower_boundary_i[i] = upper_boundary[i]
        lower_i, upper_i = generate_grids(lower_boundary_i, upper_boundary, grid_size_i)
        lower = torch.cat((lower, lower_i), dim=0)
        upper = torch.cat((upper, upper_i), dim=0)
    return lower, upper


def pgd_attack(
    x0, f, eps, steps=10, lower_boundary=None, upper_boundary=None, direction="maximize"
):
    """
    Use adversarial attack (PGD) to find violating points.
    Args:
      x0: initialization points, in [batch, state_dim].
      f: function f(x) to find the worst case x to maximize.
      eps: perturbation added to x0.
      steps: number of pgd steps.
      lower_boundary: absolute lower bounds of x.
      upper_boundary: absolute upper bounds of x.
    """
    # Set all parameters without gradient, this can speedup things significantly
    grad_status = {}
    try:
        for p in f.parameters():
            grad_status[p] = p.requires_grad
            p.requires_grad_(False)
    except:
        pass

    step_size = eps / steps * 2
    noise = torch.randn_like(x0) * step_size
    if lower_boundary is not None:
        lower_boundary = torch.max(lower_boundary, x0 - eps)
    else:
        lower_boundary = x0 - eps
    if upper_boundary is not None:
        upper_boundary = torch.min(upper_boundary, x0 + eps)
    else:
        upper_boundary = x0 + eps
    x = x0.detach().clone().requires_grad_()
    # Save the best x and best loss.
    best_x = torch.clone(x).detach().requires_grad_(False)
    fill_value = float("-inf") if direction == "maximize" else float("inf")
    best_loss = torch.full(
        size=(x.size(0),),
        requires_grad=False,
        fill_value=fill_value,
        device=x.device,
        dtype=x.dtype,
    )
    for i in range(steps):
        output = f(x).squeeze(1)
        # output = torch.clamp(f(x).squeeze(1), max=0)
        output.mean().backward()
        if direction == "maximize":
            improved_mask = output >= best_loss
        else:
            improved_mask = output <= best_loss
        best_x[improved_mask] = x[improved_mask]
        best_loss[improved_mask] = output[improved_mask]
        # print(f'step = {i}', output.view(-1).detach())
        # print(x.detach(), best_x)
        noise = torch.randn_like(x0) * step_size / (i + 1)
        if direction == "maximize":
            x = (
                (
                    torch.clamp(
                        x + torch.sign(x.grad) * step_size + noise,
                        min=lower_boundary,
                        max=upper_boundary,
                    )
                )
                .detach()
                .requires_grad_()
            )
        else:
            x = (
                (
                    torch.clamp(
                        x - torch.sign(x.grad) * step_size + noise,
                        min=lower_boundary,
                        max=upper_boundary,
                    )
                )
                .detach()
                .requires_grad_()
            )

    # restore the gradient requirement for model parameters
    try:
        for p in f.parameters():
            p.requires_grad_(grad_status[p])
    except:
        pass
    return best_x


def construct_bounded_lyapunov(
    lyaloss, lower_limit, upper_limit, grid_size, eps_ratio=1.0
):
    lower, upper = generate_grids(lower_limit, upper_limit, grid_size)
    device = lower_limit.device
    lower.to(device)
    upper.to(device)
    x = (lower + upper) / 2.0
    # Use auto_LiRPA library to bound the Lyapunov loss.
    bounded_lyapunov = BoundedModule(
        lyaloss,
        x,
        bound_opts={"conv_mode": "matrix", "sparse_intermediate_bounds": False},
        device=device,
    )
    bounded_state = construct_bounded_state(lower, upper, eps_ratio)
    return bounded_lyapunov, bounded_state, lower, upper


def construct_bounded_state(lower, upper, eps_ratio):
    x = (lower + upper) / 2.0
    eps_ratio = 0.5 + eps_ratio / 2.0
    ptb = PerturbationLpNorm(
        norm=np.inf,
        eps=None,
        x_L=eps_ratio * lower + (1.0 - eps_ratio) * upper,
        x_U=(1.0 - eps_ratio) * lower + eps_ratio * upper,
    )
    bounded_state = BoundedTensor(x, ptb)
    return bounded_state


class IbpLossReturn:
    def __init__(self, output, loss, unsatisfied, max_violation):
        self.output = output
        self.loss = loss
        self.unsatisfied = unsatisfied
        self.max_violation = max_violation


def compute_ibp_loss(
    bounded_lyapunov,
    bounded_state,
    ibp_ratio: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    grid_size: torch.Tensor,
) -> IbpLossReturn:
    device = lower_limit.device
    if ibp_ratio > 0 and bounded_lyapunov is not None:
        lb, _ = bounded_lyapunov.compute_bounds(
            x=(bounded_state,), method="IBP", bound_upper=False
        )
        bound_output = torch.clamp(-lb, min=0.0)
        bound_loss = bound_output.max()
        unsatisfied_bounds = (lb < 0).abs().sum().item()
        max_violation = bound_output.max().item()
    else:
        # skip bound computation
        grid_lower, grid_upper = generate_grids(lower_limit, upper_limit, grid_size)
        bound_loss = torch.tensor(0.0, device=device)
        unsatisfied_bounds = 0
        max_violation = 0
        bound_output = torch.zeros(size=(grid_lower.size(0), 0))
    return IbpLossReturn(bound_output, bound_loss, unsatisfied_bounds, max_violation)


class SampleLossReturn:
    def __init__(self, loss, unsatisfied, max_violation):
        self.loss = loss
        self.unsatisfied = unsatisfied
        self.max_violation = max_violation


def compute_sample_loss(
    lyaloss, x_samples: torch.Tensor, ratio: float
) -> SampleLossReturn:
    device = x_samples.device
    if ratio > 0 and lyaloss is not None:
        lya_sample = lyaloss(x_samples)
        sample_output = torch.clamp(-lya_sample, min=0.0)
        sample_loss = sample_output.sum()  # max()
        max_sample_violation = sample_output.max().item()
        unsatisfied_sample = (lya_sample < 0).abs().sum().item()
    else:
        max_sample_violation = unsatisfied_sample = 0
        sample_loss = torch.tensor(0, device=device)
    return SampleLossReturn(sample_loss, unsatisfied_sample, max_sample_violation)


class CleanLossReturn:
    def __init__(self, x, loss, unsatisfied, max_violation):
        self.x = x
        self.loss = loss
        self.unsatisfied = unsatisfied
        self.max_violation = max_violation


def compute_clean_loss(
    lyaloss, num_samples: int, limit: torch.Tensor, clean_ratio: float
) -> CleanLossReturn:
    device = limit.device
    x_dim = limit.numel()
    clean_x = (
        (
            torch.rand((num_samples, x_dim), device=device)
            - torch.full((x_dim,), 0.5, device=device)
        )
        * limit
        * 2
    )
    sample_ret = compute_sample_loss(lyaloss, clean_x, clean_ratio)
    return CleanLossReturn(
        clean_x, sample_ret.loss, sample_ret.unsatisfied, sample_ret.max_violation
    )


class AdvLossReturn:
    def __init__(self, loss, unsatisfied, max_violation, x_adv):
        self.loss = loss
        self.unsatisfied = unsatisfied
        self.max_violation = max_violation
        self.x_adv = x_adv


def compute_adv_loss(
    lyaloss,
    adv_ratio: float,
    eps: float,
    clean_x: torch.Tensor,
    pgd_steps: int,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    limit: torch.Tensor,
    adv_l1_margin: float,
    goal_state: torch.Tensor,
) -> AdvLossReturn:
    """
    Args:
    adv_l1_margin: we add a margin to the loss as
    -lyaloss(x) + adv_l1_margin*|x - goal_state|₁
    """
    device = lower_limit.device
    if adv_ratio > 0 and lyaloss is not None:
        if eps > 0:
            adv_x = pgd_attack(
                clean_x,
                lyaloss,
                eps=limit * 1 * eps,
                steps=pgd_steps,
                lower_boundary=lower_limit,
                upper_boundary=upper_limit,
                direction="minimize",
            ).detach()
        else:
            adv_x = clean_x
        adv_lya = lyaloss(adv_x)
        adv_output = torch.clamp(
            -adv_lya
            + adv_l1_margin * torch.norm(adv_x - goal_state, p=1, dim=1, keepdim=True),
            min=0.0,
        )
        max_adv_violation = adv_output.max().item()
        unsatisfied_adv = (adv_lya < 0).abs().sum().item()
        adv_loss = adv_output.sum()
    else:
        max_adv_violation = unsatisfied_adv = 0
        adv_loss = torch.tensor(0.0, device=device)
        adv_x = clean_x
    return AdvLossReturn(adv_loss, unsatisfied_adv, max_adv_violation, adv_x)


def print_progress(
    derivative_lyaloss: lyapunov.LyapunovDerivativeLoss,
    positivity_lyaloss: lyapunov.LyapunovPositivityLoss,
    iter: int,
    derivative_ibp_ret: IbpLossReturn,
    positivity_ibp_ret: IbpLossReturn,
    derivative_clean_ret: CleanLossReturn,
    positivity_clean_ret: CleanLossReturn,
    derivative_adv_ret: AdvLossReturn,
    positivity_adv_ret: AdvLossReturn,
    logger: logging.Logger,
    elapsed_time: typing.Optional[float] = None,
):
    # Check how large the Lyapunov function is. Make sure it is not converging to the trivial solution of 0.
    # Check how large the weights are, make sure they are not 0.
    total_elements_lya = sum(
        p.numel() for p in derivative_lyaloss.lyapunov.parameters()
    )
    weight_l1_lya = sum(
        p.abs().sum().item() for p in derivative_lyaloss.lyapunov.parameters()
    ) / (total_elements_lya + 1e-5)
    total_elements_controller = sum(
        p.numel() for p in derivative_lyaloss.controller.parameters()
    )
    weight_l1_controller = sum(
        p.abs().sum().item() for p in derivative_lyaloss.controller.parameters()
    ) / (total_elements_controller + 1e-5)

    # IBP loss stats.
    print_msg = f"iter={iter}, adv_loss={derivative_adv_ret.loss.item():.7f}, max_adv_vio={derivative_adv_ret.max_violation:.7f}, unsat_adv={derivative_adv_ret.unsatisfied:3d}, "
    if elapsed_time is not None:
        print_msg += "elapsed_time={elapsed_time}, "
    # Model weights and output stats.
    print_msg += f"cntl_w={weight_l1_controller:.5f}, lya_w={weight_l1_lya:.5f}, "
    if positivity_lyaloss is not None:
        print_msg += f"bound_loss_positivity={positivity_ibp_ret.loss.item():.7f}, "
        print_msg += f"clean_loss_positivity={positivity_clean_ret.loss.item():.7f}, max_clean_vio_positivity={positivity_clean_ret.max_violation:.7f}, unsat_clean_positivity={positivity_clean_ret.unsatisfied:3d}, "
        print_msg += f"adv_loss_positivity={positivity_adv_ret.loss.item():.7f}, max_adv_vio={positivity_adv_ret.max_violation:.7f}, unsat_adv={positivity_adv_ret.unsatisfied:3d}"
    logger.info(print_msg)


@dataclass
class BatchTrainLyapunovReturn:
    buffer_loss: float
    derivative_ibp_ret: IbpLossReturn
    positivity_ibp_ret: IbpLossReturn
    derivative_buffer_ret: SampleLossReturn
    positivity_batch_ret: SampleLossReturn
    l1_loss: float
    roa_loss: float
    epoch: int
    derivative_buffer_ret_init: SampleLossReturn


@dataclass
class RoaRegulizerReturn:
    loss: torch.Tensor
    Vmin_boundary: typing.Optional[torch.Tensor]
    Vmax_boundary: typing.Optional[torch.Tensor]


def roa_regulizer(
    lyap: lyapunov.NeuralNetworkLyapunov,
    Vmin_x_boundary: torch.Tensor,
    Vmin_x_boundary_weight: float,
    Vmax_x_boundary: torch.Tensor,
    Vmax_x_boundary_weight: float,
) -> RoaRegulizerReturn:
    """
    Compute the regulizer cost that encourages a larger certified ROA.

    Since the certified ROA is the sub-level set {x | V(x) ≤ ρ}
    where ρ = min_{x∈∂ℬ} V(x)
    ℬ is the verified region and ∂ℬ is its boundary, we compute the following regulizer cost
    -weight1 * min_{x∈ Vmin_x_boundary}V(x)
    +weight2 * max_{x∈ Vmax_x_boundary}V(x)

    return:
      (loss, Vmin, Vmax)
    """
    raise DeprecationWarning("roa_regulizer is outdated.")
    loss = torch.tensor(0.0, device=Vmin_x_boundary.device)
    if Vmin_x_boundary_weight != 0:
        Vmin_boundary = torch.min(lyap(Vmin_x_boundary))
        loss -= Vmin_x_boundary_weight * Vmin_boundary
    else:
        Vmin_boundary = None
    if Vmax_x_boundary_weight != 0:
        Vmax_boundary = torch.max(lyap(Vmax_x_boundary))
        loss += Vmax_x_boundary_weight * Vmax_boundary
    else:
        Vmax_boundary = None
    return RoaRegulizerReturn(loss, Vmin_boundary, Vmax_boundary)


def calc_candidate_roa_regulizer(
    lyap_nn: lyapunov.NeuralNetworkLyapunov,
    rho: torch.Tensor,
    x: typing.Optional[torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """
    Compute weight * sum(max(V(x)/rho - 1, 0))
    """
    if x is not None:
        q = lyap_nn(x) / rho - 1
        return weight * torch.nn.functional.relu(q).mean()
    else:
        return torch.tensor(0.0)


def lipschitz_regularizer(f, x_samples):
    f0 = f(x_samples)
    idx = torch.arange(x_samples.shape[0], 0, -1) - 1
    x1 = x_samples.clone()[idx]
    f1 = f0.clone()[idx]
    C = torch.norm(f1 - f0, p=2, dim=1) / torch.norm(x1 - x_samples, p=2, dim=1)
    return C.mean()


def update_x_boundary_dataset(
    Vmin_x_boundary_weight: float,
    V_decrease_within_roa: bool,
    Vmin_x_boundary: torch.Tensor,
    derivative_lyaloss: lyapunov.LyapunovDerivativeLoss,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    num_samples_per_boundary: int,
    Vmin_x_pgd_buffer_size: int,
) -> torch.Tensor:
    if Vmin_x_boundary_weight != 0 or V_decrease_within_roa:
        limit = (upper_limit - lower_limit) / 2
        Vmin_x_boundary = torch.cat(
            (
                Vmin_x_boundary,
                calc_V_extreme_on_boundary_pgd(
                    derivative_lyaloss.lyapunov,
                    lower_limit,
                    upper_limit,
                    num_samples_per_boundary,
                    eps=limit,
                    steps=100,
                    direction="minimize",
                ),
            ),
            dim=0,
        )
        if Vmin_x_boundary.shape[0] > Vmin_x_pgd_buffer_size:
            # Now sort V in ascending order.
            with torch.no_grad():
                (_, indices) = torch.sort(
                    derivative_lyaloss.lyapunov(Vmin_x_boundary).squeeze(1)
                )
                # Only pick the boundary states with small V values.
                Vmin_x_boundary = Vmin_x_boundary[indices[:Vmin_x_pgd_buffer_size]]
        if V_decrease_within_roa:
            derivative_lyaloss.x_boundary = Vmin_x_boundary
    return Vmin_x_boundary


def update_adv_dataset(
    old_buffer: torch.Tensor, new_adv_x: torch.Tensor, loss, buffer_size: int
) -> torch.Tensor:
    """
    The violation of a state x is -loss(x)
    We keep the state with the largest violation
    """
    new_buffer = torch.cat((new_adv_x, old_buffer), dim=0)
    if new_buffer.shape[0] > buffer_size:
        # Sort the violation, only retain the adversarial states with the larger violation.
        with torch.no_grad():
            violation_on_buffer = -loss(new_buffer)
            (_, sorted_indices) = torch.sort(
                violation_on_buffer.squeeze(1), descending=True
            )
            new_buffer = new_buffer[sorted_indices[:buffer_size]]
    return new_buffer


def compute_loss_on_dataset(
    derivative_lyaloss: lyapunov.LyapunovDerivativeLoss,
    positivity_lyaloss: typing.Optional[lyapunov.LyapunovPositivityLoss],
    derivative_x_samples: torch.Tensor,
    derivative_sample_ratio: float,
    positivity_x_samples: torch.Tensor,
    positivity_sample_ratio: float,
    derivative_ibp_ret: typing.Optional[IbpLossReturn],
    derivative_ibp_ratio: float,
    positivity_ibp_ret: typing.Optional[IbpLossReturn],
    positivity_ibp_ratio: float,
) -> typing.Tuple[torch.Tensor, SampleLossReturn, typing.Optional[SampleLossReturn]]:
    """
    Compute the Lyapunov loss (derivative and positivity loss) on the dataset.
    Note that this does NOT include the regularization loss or the observer loss.
    """
    derivative_buffer_ret = compute_sample_loss(
        derivative_lyaloss, derivative_x_samples, derivative_sample_ratio
    )
    buffer_loss = derivative_buffer_ret.loss * derivative_sample_ratio
    if derivative_ibp_ret is not None:
        buffer_loss += derivative_ibp_ret.loss * derivative_ibp_ratio
    if positivity_lyaloss is not None:
        positivity_buffer_ret = compute_sample_loss(
            positivity_lyaloss, positivity_x_samples, positivity_sample_ratio
        )
        buffer_loss += positivity_buffer_ret.loss * positivity_sample_ratio
        if positivity_ibp_ret is not None:
            buffer_loss += positivity_ibp_ret.loss * positivity_ibp_ratio
    else:
        positivity_buffer_ret = None
    return buffer_loss, derivative_buffer_ret, positivity_buffer_ret


def batch_train_lyapunov(
    derivative_lyaloss: lyapunov.LyapunovDerivativeLoss,
    positivity_lyaloss: lyapunov.LyapunovPositivityLoss,
    observer_loss: lyapunov.ObserverLoss,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    grid_size: torch.Tensor,
    ibp_eps: float,
    derivative_x_samples: torch.Tensor,
    positivity_x_samples: torch.Tensor,
    observer_x_samples: torch.Tensor,
    batch_size: int,
    epochs: int,
    derivative_ibp_ratio: float,
    derivative_sample_ratio: float,
    positivity_ibp_ratio: float,
    positivity_sample_ratio: float,
    lr: float,
    weight_decay: float,
    l1_reg: float,
    observer_ratio: float,
    num_samples_per_boundary: int,
    Vmin_x_boundary: torch.Tensor,
    Vmin_x_boundary_weight: float,
    Vmin_x_pgd_buffer_size: int,
    Vmax_x_boundary: torch.Tensor,
    Vmax_x_boundary_weight: float,
    candidate_roa_states: typing.Optional[torch.Tensor],
    candidate_roa_states_weight: float,
    hard_max: bool,
    V_decrease_within_roa: bool,
    update_Vmin_boundary_per_epoch: bool,
    lr_controller: float,
    lr_scheduler: bool,
    train_clf: bool,
    logger: logging.Logger,
    always_candidate_roa_regulizer: bool,
) -> BatchTrainLyapunovReturn:
    """
    Minimize the Lyapunov function loss
    c1 * ibp_loss + c2 * sample_loss
    + l1_reg * |θ|
    + roa_regulizer()
    through batched gradient descent.

    Note that we DO NOT take mini batch on the positivity_x_samples. Instead I
    will compute the loss on all positivity_x_samples. The reason is that
    usually positivity_x_samples has a small size.

    Args:
      always_candidate_roa_regulizer: If set to True, then we always add the
        candidate_roa_regulizer; otherwise we only impose this regularization
        when the Lyapuov condition violation is non-zero.
    """
    params_dict = [{"params": list(derivative_lyaloss.lyapunov.parameters()), "lr": lr}]
    if not train_clf:
        params_dict.append(
            {
                "params": list(derivative_lyaloss.controller.parameters()),
                "lr": lr_controller,
            }
        )
    if observer_loss is not None:
        params_dict.append(
            {
                "params": list(derivative_lyaloss.observer.parameters()),
                "lr": lr / 10,
            }
        )
    params = []
    for dict in params_dict:
        params += dict["params"]
    optimizer = torch.optim.Adam(params_dict, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.3), int(epochs * 0.6)], gamma=0.5
    )

    device = derivative_x_samples.device
    derivative_dataset = torch.utils.data.TensorDataset(derivative_x_samples)
    derivative_dataloader = torch.utils.data.DataLoader(
        derivative_dataset, batch_size=batch_size, shuffle=True
    )
    total_elements_lya = sum(
        p.numel() for p in derivative_lyaloss.lyapunov.parameters()
    )
    if not train_clf:
        total_elements_con = sum(
            p.numel() for p in derivative_lyaloss.controller.parameters()
        )
    if observer_loss is not None:
        total_elements_obs = sum(
            p.numel() for p in derivative_lyaloss.observer.parameters()
        )
    if derivative_lyaloss.x_boundary is not None:
        logger.info(
            f"rho is {derivative_lyaloss.get_rho().item()}, dataset size={derivative_lyaloss.x_boundary.shape[0]}"
        )

    called_optimizer_step = False
    derivative_loss_init = compute_sample_loss(
        derivative_lyaloss, derivative_x_samples, derivative_sample_ratio
    )

    if derivative_ibp_ratio > 0:
        (
            bounded_lyapunov,
            bounded_state,
            grid_lower,
            grid_upper,
        ) = construct_bounded_lyapunov(
            derivative_lyaloss, lower_limit, upper_limit, grid_size, ibp_eps
        )
    else:
        bounded_lyapunov = None
        bounded_state = None

    if derivative_ibp_ratio > 0:
        (
            bounded_lyapunov,
            bounded_state,
            grid_lower,
            grid_upper,
        ) = construct_bounded_lyapunov(
            derivative_lyaloss, lower_limit, upper_limit, grid_size, ibp_eps
        )
    else:
        bounded_lyapunov = None
        bounded_state = None

    for i in range(epochs):
        if update_Vmin_boundary_per_epoch:
            Vmin_x_boundary = update_x_boundary_dataset(
                Vmin_x_boundary_weight,
                V_decrease_within_roa,
                Vmin_x_boundary,
                derivative_lyaloss,
                lower_limit,
                upper_limit,
                num_samples_per_boundary,
                Vmin_x_pgd_buffer_size,
            )
        # Compute the initial derivative loss on the entire dataset
        for derivative_x in derivative_dataloader:
            positivity_ibp_ret = compute_ibp_loss(
                None, None, positivity_ibp_ratio, lower_limit, upper_limit, grid_size
            )
            derivative_ibp_ret = compute_ibp_loss(
                bounded_lyapunov,
                bounded_state,
                derivative_ibp_ratio,
                lower_limit,
                upper_limit,
                grid_size,
            )
            # Compute the sample loss on the batch
            derivative_batch_ret = compute_sample_loss(
                derivative_lyaloss, derivative_x[0], derivative_sample_ratio
            )
            positivity_batch_ret = compute_sample_loss(
                positivity_lyaloss, positivity_x_samples, positivity_sample_ratio
            )

            loss = (
                derivative_ibp_ret.loss * derivative_ibp_ratio
                + derivative_batch_ret.loss * derivative_sample_ratio
            )
            if positivity_lyaloss is not None:
                loss += (
                    positivity_ibp_ret.loss * positivity_ibp_ratio
                    + positivity_batch_ret.loss * positivity_sample_ratio
                )

            rho = derivative_lyaloss.get_rho()
            if loss > 0 or always_candidate_roa_regulizer:
                candidate_roa_regulizer = calc_candidate_roa_regulizer(
                    derivative_lyaloss.lyapunov,
                    rho,
                    candidate_roa_states,
                    candidate_roa_states_weight,
                ).to(device)
                loss += candidate_roa_regulizer
            if loss > 0:
                lya_params = torch.cat(
                    [
                        p.contiguous().view(-1)
                        for p in derivative_lyaloss.lyapunov.parameters()
                    ]
                )
                l1_loss = torch.norm(lya_params, 1) / total_elements_lya
                if not train_clf:
                    con_params = torch.cat(
                        [p.view(-1) for p in derivative_lyaloss.controller.parameters()]
                    )
                    l1_loss += torch.norm(con_params, 1) / total_elements_con / 10
                # l1_loss = lipschitz_regularizer(derivative_lyaloss.lyapunov, derivative_x[0]) + \
                #     lipschitz_regularizer(derivative_lyaloss.controller, derivative_x[0])
                if observer_loss is not None:
                    obs_params = torch.cat(
                        [p.view(-1) for p in derivative_lyaloss.observer.parameters()]
                    )
                    l1_loss += torch.norm(obs_params, 1) / total_elements_obs / 10
                if observer_loss is not None and observer_ratio > 0:
                    loss += observer_ratio * observer_loss(observer_x_samples).mean()
                loss += l1_reg * l1_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()
                called_optimizer_step = True
            if lr_scheduler:
                scheduler.step()
        # Compute the loss on the entire buffer.
        # We have already computed the positivity loss on the entire buffer,
        # so only need to recompute the derivative loss.
        rho = derivative_lyaloss.get_rho()
        with torch.no_grad():
            candidate_roa_regulizer = calc_candidate_roa_regulizer(
                derivative_lyaloss.lyapunov,
                rho,
                candidate_roa_states,
                candidate_roa_states_weight,
            ).to(device)
            (
                buffer_loss,
                derivative_buffer_ret,
                positivity_buffer_ret,
            ) = compute_loss_on_dataset(
                derivative_lyaloss,
                positivity_lyaloss,
                derivative_x_samples,
                derivative_sample_ratio,
                positivity_x_samples,
                positivity_sample_ratio,
                derivative_ibp_ret,
                derivative_ibp_ratio,
                positivity_ibp_ret,
                positivity_ibp_ratio,
            )
            weight_l1_lya = (
                sum(
                    p.abs().sum().item()
                    for p in derivative_lyaloss.lyapunov.parameters()
                )
                / (total_elements_lya)
                * l1_reg
            )
            if observer_loss is not None:
                obs_loss = (
                    observer_ratio * observer_loss(observer_x_samples).mean().item()
                )
            else:
                obs_loss = 0
            print_msg = (
                f"epoch {i}, adv_sample={derivative_buffer_ret.unsatisfied}, "
                + f"adv_loss={buffer_loss.item()}, obs={obs_loss}, l1={weight_l1_lya}, "
            )
            if candidate_roa_regulizer is not None:
                print_msg += (
                    f"candidate_roa_regulizer={candidate_roa_regulizer.item()}, "
                )
            if called_optimizer_step:
                logger.info(print_msg)
            if buffer_loss == 0:
                break
    return BatchTrainLyapunovReturn(
        buffer_loss.item(),
        derivative_ibp_ret,
        positivity_ibp_ret,
        derivative_buffer_ret,
        positivity_buffer_ret,
        weight_l1_lya,
        candidate_roa_regulizer.item(),
        epoch=i,
        derivative_buffer_ret_init=derivative_loss_init,
    )


@dataclass
class TrainLyapunovWithBufferReturn:
    derivative_adv_samples: torch.Tensor
    x_min_boundary: torch.Tensor


def train_lyapunov_with_buffer(
    *,
    derivative_lyaloss: lyapunov.LyapunovDerivativeLoss,
    positivity_lyaloss: typing.Optional[lyapunov.LyapunovPositivityLoss],
    observer_loss: typing.Optional[lyapunov.ObserverLoss],
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    grid_size: torch.Tensor,
    learning_rate=0.001,
    lr_controller=0.001,
    weight_decay=0.0,
    max_iter=10000,
    enable_wandb=False,
    derivative_ibp_ratio=0.0,
    derivative_sample_ratio=1.0,
    positivity_ibp_ratio=0.0,
    positivity_sample_ratio=1.0,
    save_best_model=None,
    pgd_steps=10,
    buffer_size=1000,
    batch_size=100,
    epochs=20,
    samples_per_iter=100,
    l1_reg=1e-3,
    observer_ratio=1e-3,
    num_samples_per_boundary=1000,
    V_decrease_within_roa: bool = False,
    Vmin_x_boundary_weight=0.0,
    Vmax_x_boundary_weight=0.0,
    candidate_roa_states: typing.Optional[torch.Tensor] = None,
    candidate_roa_states_weight: float = 0.0,
    hard_max: bool = True,
    lr_scheduler=False,
    Vmin_x_pgd_buffer_size=500000,
    update_Vmin_boundary_per_epoch=False,
    derivative_x_buffer=None,
    Vmin_x_pgd=None,
    train_clf: bool = False,
    logger: logging.Logger = None,
    always_candidate_roa_regulizer: bool = False,
) -> TrainLyapunovWithBufferReturn:
    """
    We train the Lyapunov and controller iteratively.
    In each iteration:
    1. We first find the adversarial states through PGD attack.
    2. We append the adversarial states to a buffer.
    3. We minimize the loss function through mini-batch gradient descent on that buffer.
    Args:
      buffer_size: The maximal number of samples in the buffer. The oldest
      buffers are discarded first.
      batch_size: The size of the mini-batch in step 3.
      epochs: Number of epochs in step 3.
      samples_per_iter: Number of samples in the PGD attack in step 1.
      always_candidate_roa_regulizer: If set to True, then we always add the
        candidate_roa_regulizer; otherwise we only impose this regularization
        when the Lyapuov condition violation is non-zero.
    """
    limit = (upper_limit - lower_limit) / 2.0
    nx = derivative_lyaloss.lyapunov.x_dim
    device = lower_limit.device
    dtype = lower_limit.dtype
    if derivative_x_buffer is None:
        derivative_x_buffer = torch.empty((0, nx), device=device)
    best_loss = np.inf

    # Find argmin(V(x_boundary)) through PGD attack on the boundary of the box.
    if Vmin_x_pgd is None:
        Vmin_x_pgd = torch.empty((0, nx), device=device)

    # fig, ax = plt.subplots(1, 2)
    start_time = time.time()
    if logger is None:
        logger = logging.getLogger(__name__)
    for i in range(max_iter):
        logger.info(f"iter={i}")

        Vmin_x_pgd = update_x_boundary_dataset(
            Vmin_x_boundary_weight,
            V_decrease_within_roa,
            Vmin_x_pgd,
            derivative_lyaloss,
            lower_limit,
            upper_limit,
            num_samples_per_boundary,
            Vmin_x_pgd_buffer_size,
        )

        # First find the adversarial states through PGD attack.

        # TODO(hongkai.dai): figure out a better way to get the clean_x. Should
        # I start from the adversarial states in the previous iteration?
        clean_x = (
            (
                torch.rand((samples_per_iter, nx), device=device, dtype=dtype)
                - torch.full((nx,), 0.5, device=device, dtype=dtype)
            )
            * limit
            * 2
        )
        derivative_adv_x = pgd_attack(
            clean_x,
            derivative_lyaloss,
            eps=limit,
            steps=pgd_steps,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        derivative_x_buffer = update_adv_dataset(
            derivative_x_buffer, derivative_adv_x, derivative_lyaloss, buffer_size
        )
        if positivity_lyaloss is not None:
            positivity_adv_x = pgd_attack(
                clean_x,
                positivity_lyaloss,
                eps=limit,
                steps=pgd_steps,
                lower_boundary=lower_limit,
                upper_boundary=upper_limit,
                direction="minimize",
            ).detach()
        else:
            positivity_adv_x = clean_x

        # Find maximal and minimal of V on the boundary of the verified region through PGD attacks.
        Vmax_x_pgd = (
            torch.empty((0, nx), device=device)
            if Vmax_x_boundary_weight == 0
            else calc_V_extreme_on_boundary_pgd(
                derivative_lyaloss.lyapunov,
                lower_limit,
                upper_limit,
                num_samples_per_boundary,
                eps=limit,
                steps=100,
                direction="maximize",
            )
        )

        ibp_eps = 1.0
        batch_train_lyapunov_ret = batch_train_lyapunov(
            derivative_lyaloss,
            positivity_lyaloss,
            observer_loss,
            lower_limit,
            upper_limit,
            grid_size,
            ibp_eps,
            derivative_x_buffer,
            positivity_adv_x,
            clean_x,
            batch_size,
            epochs,
            derivative_ibp_ratio,
            derivative_sample_ratio,
            positivity_ibp_ratio,
            positivity_sample_ratio,
            learning_rate,
            weight_decay,
            l1_reg,
            observer_ratio,
            num_samples_per_boundary,
            Vmin_x_pgd,
            Vmin_x_boundary_weight,
            Vmin_x_pgd_buffer_size,
            Vmax_x_pgd,
            Vmax_x_boundary_weight,
            candidate_roa_states,
            candidate_roa_states_weight,
            hard_max,
            V_decrease_within_roa,
            update_Vmin_boundary_per_epoch,
            lr_controller,
            lr_scheduler,
            train_clf,
            logger,
            always_candidate_roa_regulizer,
        )
        elapsed_time = time.time() - start_time
        logger.info(f"elapsed time = {elapsed_time}")
        loss = batch_train_lyapunov_ret.buffer_loss
        if loss <= best_loss:
            best_loss = loss
            if save_best_model is not None:
                model_dict = {
                    "state_dict": derivative_lyaloss.state_dict(),
                    "lower_limit": lower_limit,
                    "upper_limit": upper_limit,
                    "kappa": derivative_lyaloss.kappa,
                    "ibp_ratio_derivative": derivative_ibp_ratio,
                    "sample_ratio_derivative": derivative_sample_ratio,
                    "ibp_ratio_positivity": positivity_ibp_ratio,
                    "sample_ratio_positivity": positivity_sample_ratio,
                    "x_boundary": derivative_lyaloss.x_boundary,
                }
                if hasattr(derivative_lyaloss, "get_rho"):
                    model_dict["rho"] = derivative_lyaloss.get_rho()
                torch.save(
                    model_dict,
                    save_best_model,
                )
        if enable_wandb:
            wandb.log(
                {
                    "buffer_loss": batch_train_lyapunov_ret.buffer_loss,
                    "l1_loss": batch_train_lyapunov_ret.l1_loss,
                    "roa_loss": batch_train_lyapunov_ret.roa_loss,
                    "epoch": batch_train_lyapunov_ret.epoch,
                    "initial_buffer_loss": batch_train_lyapunov_ret.derivative_buffer_ret_init.loss,
                    "initial_buffer_violations": batch_train_lyapunov_ret.derivative_buffer_ret_init.unsatisfied,
                }
            )
    return TrainLyapunovWithBufferReturn(
        derivative_adv_samples=derivative_x_buffer, x_min_boundary=Vmin_x_pgd
    )


def plot_adversarial_samples(ax, x_adv, x_max, e_max, file_name):
    if isinstance(x_adv, torch.Tensor):
        x_adv = x_adv.cpu().detach().numpy()
        x_max = x_max.cpu().detach().numpy()
        e_max = e_max.cpu().detach().numpy()
    x_min = -x_max
    e_min = -e_max
    ax[0].plot(x_adv[:, 0], x_adv[:, 1], "r.")
    ax[0].set_xlim([x_min[0], x_max[0]])
    ax[0].set_ylim([x_min[1], x_max[1]])
    ax[0].set_xlabel("x0")
    ax[0].set_ylabel("x1")
    ax[0].axis("equal")
    ax[1].plot(x_adv[:, 2], x_adv[:, 3], "r.")
    ax[1].set_xlim([e_min[0], e_max[0]])
    ax[1].set_ylim([e_min[1], e_max[1]])
    ax[1].set_xlabel("e0")
    ax[1].set_ylabel("e1")
    ax[1].axis("equal")
    plt.savefig(file_name)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def calc_V_extreme_on_boundary_pgd(
    lyap: lyapunov.NeuralNetworkLyapunov,
    lower_boundary: torch.Tensor,
    upper_boundary: torch.Tensor,
    num_samples_per_boundary: int,
    eps: float,
    steps: int,
    direction="maximize",
) -> torch.Tensor:
    """
    Find max V(x) (or min V(x)) over x on the boundary of the box through PGD
    attack.

    Args:
      eps: See eps in pgd_attack().
      steps: See steps in pgd_attack().
      direction: where we maximize or minimize V.
    return x_pgd The sample points x after the pgd attack.
    """
    x_pgd = []
    device = lower_boundary.device

    # Loop through each face on the boundary
    for i in range(lower_boundary.shape[0]):
        # Generate samples with the i'th coordinate fixed.
        range_on_boundary = upper_boundary - lower_boundary
        range_on_boundary[i] = 0.0
        boundary_center = (lower_boundary + upper_boundary) / 2
        # Lower face
        boundary_center[i] = lower_boundary[i]
        x_samples_on_boundary = (
            torch.rand(
                (num_samples_per_boundary, lower_boundary.shape[0]), device=device
            )
            - torch.full(
                (num_samples_per_boundary, lower_boundary.shape[0]), 0.5, device=device
            )
        ) * range_on_boundary + boundary_center
        upper_boundary_i = upper_boundary.clone()
        upper_boundary_i[i] = lower_boundary[i]
        x_pgd_lower = pgd_attack(
            x_samples_on_boundary,
            lyap,
            eps,
            steps,
            lower_boundary=lower_boundary,
            upper_boundary=upper_boundary_i,
            direction=direction,
        )
        x_pgd.append(x_pgd_lower.detach())
        # Upper face
        boundary_center[i] = upper_boundary[i]
        x_samples_on_boundary = (
            torch.rand(
                (num_samples_per_boundary, lower_boundary.shape[0]), device=device
            )
            - torch.full(
                (num_samples_per_boundary, lower_boundary.shape[0]), 0.5, device=device
            )
        ) * range_on_boundary + boundary_center
        lower_boundary_i = lower_boundary.clone()
        lower_boundary_i[i] = upper_boundary[i]
        x_pgd_upper = pgd_attack(
            x_samples_on_boundary,
            lyap,
            eps,
            steps,
            lower_boundary=lower_boundary_i,
            upper_boundary=upper_boundary,
            direction=direction,
        )
        x_pgd.append(x_pgd_upper.detach())
    x_pgd = torch.cat(x_pgd, dim=0)
    return x_pgd


def get_candidate_roa_states(
    V: lyapunov.NeuralNetworkLyapunov,
    rho: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    box_scale: float,
) -> torch.Tensor:
    """
    Move from box corners towards the sub-level set {x | V(x) <= rho}.

    We apply PGD attack with to minimize max(V(x) - rho, 0)
    """
    nx = lower_limit.numel()
    device = lower_limit.device
    permute_array = [[-1, 1]] * nx
    permute_array_torch = torch.tensor(
        list(itertools.product(*permute_array)), device=device
    )
    candidate_roa_states = permute_array_torch * upper_limit * box_scale
    if rho is not None:
        eps = (upper_limit - lower_limit) / 2

        class CandidateRoaPgdLoss(torch.nn.Module):
            """
            Compute max(V(x)-rho, 0)
            """

            def __init__(self, V: lyapunov.NeuralNetworkLyapunov, rho: float):
                super().__init__()
                self.V = V
                self.rho = rho

            def forward(self, x: torch.Tensor):
                V_x = self.V(x)
                return torch.maximum(
                    V_x - self.rho, torch.zeros_like(V_x, device=x.device)
                )

        loss = CandidateRoaPgdLoss(V, rho)
        candidate_roa_states = pgd_attack(
            candidate_roa_states,
            loss,
            eps=eps,
            steps=100,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
    return candidate_roa_states


def plot_V_heatmap(
    fig,
    V,
    rho,
    lower_limit,
    upper_limit,
    nx,
    x_boundary,
    plot_idx=[0, 1],
    mode=0.0,
    V_color="k",
    V_lqr=None,
):
    device = lower_limit.device
    x_ticks = torch.linspace(
        lower_limit[plot_idx[0]], upper_limit[plot_idx[0]], 500, device=device
    )
    y_ticks = torch.linspace(
        lower_limit[plot_idx[1]], upper_limit[plot_idx[1]], 500, device=device
    )
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    if mode == "boundary":
        X = torch.ones(250000, nx, device=device) * x_boundary
    elif isinstance(mode, float):
        X = torch.ones(250000, nx, device=device) * upper_limit * mode
    X[:, plot_idx[0]] = grid_x.flatten()
    X[:, plot_idx[1]] = grid_y.flatten()

    with torch.no_grad():
        V_val = V(X)

    V_val = V_val.cpu().reshape(500, 500)
    grid_x = grid_x.cpu()
    grid_y = grid_y.cpu()

    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x, grid_y, V_val, cmap=cm.coolwarm)
    ax.contour(grid_x, grid_y, V_val, [rho], colors=V_color, linewidths=2.5)
    if V_lqr is not None:
        V_lqr_val = V_lqr(X).reshape(500, 500).cpu()
        x_pgd_boundary_min = calc_V_extreme_on_boundary_pgd(
            V_lqr,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=5000,
            eps=(upper_limit - lower_limit) / 2,
            steps=300,
            direction="minimize",
        )
        rho_lqr_roa = torch.min(V_lqr(x_pgd_boundary_min)).item()
        ax.contour(
            grid_x, grid_y, V_lqr_val, [rho_lqr_roa], colors="orange", linewidths=2.5
        )
    lower_limit = lower_limit.cpu()
    upper_limit = upper_limit.cpu()
    ax.set_xlim(lower_limit[plot_idx[0]], upper_limit[plot_idx[0]])
    ax.set_ylim(lower_limit[plot_idx[1]], upper_limit[plot_idx[1]])
    cbar = fig.colorbar(im, ax=ax)
    return ax, cbar
