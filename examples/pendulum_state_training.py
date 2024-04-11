import os
import pdb

import argparse
import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import scipy.linalg
import torch
import torch.nn as nn
import wandb

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.train_utils as train_utils

device = torch.device("cuda")
dtype = torch.float


def linearize_pendulum(pendulum_continuous: pendulum.PendulumDynamics):
    x = torch.tensor([[0.0, 0.0]])
    x.requires_grad = True
    u = torch.tensor([[0.0]])
    u.requires_grad = True
    qddot = pendulum_continuous.forward(x, u)
    A = torch.empty((2, 2))
    B = torch.empty((2, 1))
    A[0, 0] = 0
    A[0, 1] = 1
    B[0, 0] = 0
    A[1], B[1] = torch.autograd.grad(qddot[0, 0], [x, u])
    return A, B


def compute_lqr(pendulum_continuous: pendulum.PendulumDynamics):
    A, B = linearize_pendulum(pendulum_continuous)
    A_np, B_np = A.detach().numpy(), B.detach().numpy()
    Q = np.eye(2)
    R = np.eye(1) * 100
    S = scipy.linalg.solve_continuous_are(A_np, B_np, Q, R)
    K = -np.linalg.solve(R, B_np.T @ S)
    return K, S


def approximate_lqr(
    pendulum_continuous: pendulum.PendulumDynamics,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn: lyapunov.NeuralNetworkLyapunov,
    upper_limit: torch.Tensor,
    logger,
):
    K, S = compute_lqr(pendulum_continuous)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    x = (torch.rand((100000, 2), dtype=dtype, device=device) - 0.5) * 2 * upper_limit
    V = torch.sum(x * (x @ S_torch), axis=1, keepdim=True)
    u = x @ K_torch.T

    def approximate(system, system_input, target, lr, max_iter):
        optimizer = torch.optim.Adam(system.parameters(), lr=lr)
        for i in range(max_iter):
            optimizer.zero_grad()
            output = torch.nn.MSELoss()(system.forward(system_input), target)
            logger.info(f"iter {i}, loss {output.item()}")
            output.backward()
            optimizer.step()

    approximate(controller, x, u, lr=0.01, max_iter=500)
    approximate(lyapunov_nn, x, V, lr=0.01, max_iter=1000)


def plot_V(V, lower_limit, upper_limit):
    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], 50, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], 50, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    with torch.no_grad():
        V_val = V(torch.stack((grid_x, grid_y), dim=2)).squeeze(2)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(grid_x.numpy(), grid_y.numpy(), V_val.numpy())
    ax.set_xlabel(r"$\theta (rad)$")
    ax.set_ylabel(r"$\dot{\theta} (rad/s)$")
    ax.set_zlabel("V")
    return fig, ax


def plot_V_heatmap(V, lower_limit, upper_limit, rho):
    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], 1000, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], 1000, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    with torch.no_grad():
        V_val = V.forward(torch.stack((grid_x, grid_y), dim=2)).squeeze(2)

    V_val = V_val.cpu()
    grid_x = grid_x.cpu()
    grid_y = grid_y.cpu()

    lower_limit = lower_limit.cpu()
    upper_limit = upper_limit.cpu()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x, grid_y, V_val)
    ax.contour(grid_x, grid_y, V_val, [rho], colors="red")
    ax.set_xlim(lower_limit[0], upper_limit[0])
    ax.set_ylim(lower_limit[1], upper_limit[1])
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar


@hydra.main(config_path="./config", config_name="pendulum_state_training")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    train_utils.set_seed(cfg.seed)

    dt = cfg.model.dt
    pendulum_continuous = pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        pendulum_continuous,
        dt=dt,
        position_integration=dynamical_system.IntegrationMethod[
            cfg.model.position_integration
        ],
        velocity_integration=dynamical_system.IntegrationMethod[
            cfg.model.velocity_integration
        ],
    )

    controller = controllers.NeuralNetworkController(
        nlayer=cfg.model.controller_nlayer,
        in_dim=2,
        out_dim=1,
        hidden_dim=cfg.model.controller_hidden_dim,
        clip_output="clamp",
        u_lo=torch.tensor([-cfg.model.u_max]),
        u_up=torch.tensor([cfg.model.u_max]),
        x_equilibrium=pendulum_continuous.x_equilibrium,
        u_equilibrium=pendulum_continuous.u_equilibrium,
    )
    controller.eval()

    absolute_output = True
    if cfg.model.lyapunov.quadratic:
        _, S = compute_lqr(pendulum_continuous)
        S_torch = torch.from_numpy(S).type(dtype).to(device)
        R = torch.linalg.cholesky(S_torch)
        lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
            goal_state=torch.zeros(2, dtype=dtype).to(device),
            x_dim=2,
            R_rows=2,
            eps=0.01,
            R=R,
        )
        controller.load_state_dict(
            torch.load(
                os.path.join(
                    os.path.dirname(__file__), "../", cfg.model.controller_path
                )
            )
        )
    else:
        lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
            goal_state=torch.tensor([0.0, 0.0]),
            hidden_widths=cfg.model.lyapunov.hidden_widths,
            x_dim=2,
            R_rows=3,
            absolute_output=absolute_output,
            eps=0.01,
            activation=nn.LeakyReLU,
            V_psd_form=cfg.model.V_psd_form,
        )
    lyapunov_nn.eval()

    kappa = cfg.model.kappa
    rho_multiplier = cfg.model.rho_multiplier
    # Place holder for lyaloss
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=0,
        box_up=0,
        rho_multiplier=rho_multiplier,
        kappa=kappa,
        hard_max=cfg.train.hard_max,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    grid_size = torch.tensor([50, 50], device=device)
    # approximate_controller(controller_target, controller, 2, limit, 0, 0, "examples/pendulum_controller.pth", batch_size=10000, max_iter=500)
    # approximate_controller(lyapunov_target, lyapunov_nn, 2, limit, 0, 0, "examples/pendulum_lyapunov.pth", batch_size=10000, max_iter=500)
    logger = logging.getLogger(__name__)
    if cfg.approximate_lqr:
        approximate_lqr(
            pendulum_continuous, controller, lyapunov_nn, upper_limit, logger
        )
        torch.save(
            {"state_dict": derivative_lyaloss.state_dict()},
            os.path.join(os.getcwd(), "lyaloss_lqr.pth"),
        )
        return

    if cfg.model.load_lyaloss is not None:
        load_lyaloss = os.path.join(
            os.path.dirname(__file__), "../", cfg.model.load_lyaloss
        )
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(2, device=device)
        )

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )
        # wandb.config.update(cfg)

    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa

    if cfg.train.train_lyaloss:
        for n in range(len(cfg.model.limit_scale)):
            limit_scale = cfg.model.limit_scale[n]
            limit = limit_scale * torch.tensor(cfg.model.limit, device=device)
            lower_limit = -limit
            upper_limit = limit

            derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
                dynamics,
                controller,
                lyapunov_nn,
                box_lo=lower_limit,
                box_up=upper_limit,
                rho_multiplier=rho_multiplier,
                kappa=kappa,
                hard_max=cfg.train.hard_max,
            )

            if save_lyaloss:
                save_lyaloss_path = os.path.join(
                    os.getcwd(), f"lyaloss_{limit_scale}.pth"
                )
            else:
                save_lyaloss_path = None

            candidate_roa_states = limit_scale * torch.tensor(
                cfg.loss.candidate_roa_states,
                device=device,
            )

            train_utils.train_lyapunov_with_buffer(
                derivative_lyaloss=derivative_lyaloss,
                positivity_lyaloss=positivity_lyaloss,
                observer_loss=None,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                grid_size=grid_size,
                learning_rate=cfg.train.learning_rate,
                weight_decay=0.0,
                max_iter=cfg.train.max_iter,
                enable_wandb=cfg.train.wandb.enabled,
                derivative_ibp_ratio=cfg.loss.ibp_ratio_derivative,
                derivative_sample_ratio=cfg.loss.sample_ratio_derivative,
                positivity_ibp_ratio=cfg.loss.ibp_ratio_positivity,
                positivity_sample_ratio=cfg.loss.sample_ratio_positivity,
                save_best_model=save_lyaloss_path,
                pgd_steps=cfg.train.pgd_steps,
                buffer_size=cfg.train.buffer_size,
                batch_size=cfg.train.batch_size,
                epochs=cfg.train.epochs,
                samples_per_iter=cfg.train.samples_per_iter,
                l1_reg=cfg.loss.l1_reg,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                V_decrease_within_roa=V_decrease_within_roa,
                Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
                Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
                candidate_roa_states=candidate_roa_states,
                candidate_roa_states_weight=cfg.loss.candidate_roa_states_weight,
                logger=logger,
                always_candidate_roa_regulizer=cfg.loss.always_candidate_roa_regulizer,
            )

        torch.save(
            {
                "state_dict": lyapunov_nn.state_dict(),
                "rho": derivative_lyaloss.get_rho(),
            },
            os.path.join(os.getcwd(), "lyapunov_nn.pth"),
        )
    else:
        limit = cfg.model.limit_scale[-1] * torch.tensor(cfg.model.limit, device=device)
        lower_limit = -limit
        upper_limit = limit
        derivative_lyaloss.x_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            eps=limit,
            steps=100,
            direction="minimize",
        )

    # Check with pgd attack.
    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=rho_multiplier,
        kappa=0.0,
        hard_max=True,
    )
    pgd_verifier_find_counterexamples = False
    for seed in range(50):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                eps=limit,
                steps=100,
                direction="minimize",
            )
            if derivative_lyaloss.x_boundary is not None:
                derivative_lyaloss_check.x_boundary = torch.cat(
                    (x_min_boundary, derivative_lyaloss.x_boundary), dim=0
                )
        x_check_start = (
            (
                torch.rand((50000, 2), device=device)
                - torch.full((2,), 0.5, device=device)
            )
            * limit
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit,
            steps=cfg.pgd_verifier_steps,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        if max_adv_violation > 0:
            pgd_verifier_find_counterexamples = True
        logger.info(msg)
    logger.info(
        f"PGD verifier finds counter examples? {pgd_verifier_find_counterexamples}"
    )

    x0 = (torch.rand((40, 2), device=device) - 0.5) * 2 * limit
    x_traj, V_traj = models.simulate(derivative_lyaloss, 500, x0)
    plt.plot(torch.stack(V_traj).cpu().detach().squeeze().numpy())
    plt.savefig(os.path.join(os.getcwd(), "Vtraj_roa.png"))

    # pdb.set_trace()
    rho = derivative_lyaloss.get_rho().item()
    print("rho = ", rho)
    fig = plt.figure()
    train_utils.plot_V_heatmap(
        fig,
        lyapunov_nn,
        rho,
        lower_limit,
        upper_limit,
        pendulum_continuous.nx,
        derivative_lyaloss.x_boundary,
    )
    x_traj = torch.stack(x_traj).cpu().detach().squeeze().numpy()
    plt.plot(x_traj[:, :, 0], x_traj[:, :, 1])
    fig.show()
    plt.savefig(os.path.join(os.getcwd(), "V_roa.png"))

    pass


if __name__ == "__main__":
    main()
