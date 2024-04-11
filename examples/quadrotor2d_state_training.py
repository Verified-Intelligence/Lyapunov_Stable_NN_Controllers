import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
import itertools

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.quadrotor2d as quadrotor2d
import neural_lyapunov_training.train_utils as train_utils
import neural_lyapunov_training.output_train_utils as output_train_utils

device = torch.device("cuda")
dtype = torch.float


def approximate_lqr(
    quadrotor: quadrotor2d.Quadrotor2DDynamics,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn: lyapunov.NeuralNetworkLyapunov,
    logger: logging.Logger,
    limit,
):
    Q = np.diag(np.array([1, 1, 1, 10, 10, 10.0]))
    R = np.diag(np.array([10, 10.0]))
    K, S = quadrotor.lqr_control(Q, R, quadrotor.x_equilibrium, quadrotor.u_equilibrium)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)

    # We will sample x during each training iteration.
    V = lambda x: torch.sum(x * (x @ S_torch), axis=1, keepdim=True) / 50
    u = lambda x: x @ K_torch.T + quadrotor.u_equilibrium.to(device)

    def approximate(system, target_func, lr, max_iter, l1_reg=1.0):
        optimizer = torch.optim.Adam(system.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_iter
        )
        total_elements = sum(p.numel() for p in system.parameters())
        for i in range(max_iter):
            optimizer.zero_grad(set_to_none=True)
            # Sample x and compute target.
            x = (torch.rand((100000, 6), device=device) - 0.5) * limit
            y = target_func(x)
            output = torch.nn.MSELoss()(system(x), y)
            # Compute a L1 norm loss to encourage tighter IBP bounds.
            l1_loss = (
                l1_reg
                * sum(p.abs().sum() for p in system.parameters())
                / total_elements
            )
            loss = output + l1_loss
            loss.backward()
            logger.info(
                f"iter {i}, mse {output.item()}, l1 {l1_loss.item()}, loss {loss.item()}, lr {scheduler.get_last_lr()[0]:.5f}"
            )
            optimizer.step()
            scheduler.step()

    # TODO: tune L1 reg term.
    approximate(lyapunov_nn, V, lr=0.02, max_iter=5000, l1_reg=0.01)
    if (
        len(list(controller.parameters())) > 0
    ):  # Do not train if there are no parameters (e.g., fixed linear controller).
        approximate(controller, u, lr=0.05, max_iter=500, l1_reg=0.01)


def plot_V(V, lower_limit, upper_limit):
    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], 50, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], 50, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    with torch.no_grad():
        V_val = V(torch.stack((grid_x, grid_y), dim=2)).squeeze(2)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(grid_x.numpy(), grid_y.numpy(), V_val.numpy())
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_zlabel("V")
    return fig, ax


@hydra.main(config_path="./config", config_name="quadrotor2d_state_training")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    train_utils.set_seed(cfg.seed)

    dt = 0.01
    quadrotor_continuous = quadrotor2d.Quadrotor2DDynamics()
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(quadrotor_continuous, dt)

    grid_size = torch.tensor([4, 4, 6, 5, 5, 6], device=device)

    Q = np.diag(np.array([1, 1, 1, 10, 10, 10.0]))
    R = np.diag(np.array([10, 10.0]))
    K, S = quadrotor_continuous.lqr_control(
        Q, R, quadrotor_continuous.x_equilibrium, quadrotor_continuous.u_equilibrium
    )
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)

    V = (
        lambda x: torch.sum(x * (x @ S_torch), axis=1, keepdim=True) / 50
    )  # Scale V_lqr to be in [0, 10]
    u = lambda x: x @ K_torch.T + quadrotor_continuous.u_equilibrium.to(device)
    controller_target = lambda x: torch.clamp(
        u(x),
        min=torch.tensor([0, 0.0], device=device),
        max=dynamics.u_equilibrium.to(device) * 2.5,
    )

    controller = controllers.NeuralNetworkController(
        nlayer=cfg.model.controller_nlayer,
        in_dim=6,
        out_dim=2,
        hidden_dim=cfg.model.controller_hidden_dim,
        clip_output="clamp",
        u_lo=torch.tensor([0, 0.0], dtype=dtype, device=device),
        u_up=(dynamics.u_equilibrium * 2.5).to(device).to(dtype),
        x_equilibrium=(dynamics.x_equilibrium).to(device).to(dtype),
        u_equilibrium=(dynamics.u_equilibrium).to(device).to(dtype),
    )
    controller.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/quadrotor2d/controller_lqr_[8, 8].pth",
            )
        )
    )
    controller.eval()

    absolute_output = True
    R = torch.linalg.cholesky(S_torch)
    lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
        goal_state=torch.zeros(6, dtype=dtype).to(device),
        x_dim=6,
        R_rows=6,
        eps=0.01,
        R=R,
    )

    dynamics.to(device).to(dtype)
    controller.to(device).to(dtype)
    lyapunov_nn.to(device).to(dtype)

    # output_train_utils.approximate_controller(controller_target, controller, 6, limit, 0, 0, "examples/data/quadrotor2d/residue/controller_[8, 8, 8].pth", max_iter=500, lr=0.08, l1_reg=0.001)
    # output_train_utils.approximate_controller(V, lyapunov_nn, 6, limit, 0, 0, "examples/data/quadrotor2d/lyapunov_{}.pth".format(lyapunov_hidden_widths), max_iter=500, lr=0.05, l1_reg=0.01)
    logger = logging.getLogger(__name__)

    kappa = cfg.model.kappa
    hard_max = cfg.train.hard_max
    rho_multiplier = cfg.model.rho_multiplier
    # Place holder for the Lyapunov loss
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        0,
        0,
        rho_multiplier,
        kappa=kappa,
        hard_max=hard_max,
        beta=1,
        loss_weights=torch.tensor([0.5, 1, 0.5], device=device),
    )

    if cfg.approximate_lqr:
        approximate_lqr(quadrotor_continuous, controller, lyapunov_nn, logger, limit)
        torch.save(
            {"state_dict": derivative_lyaloss.state_dict()},
            os.path.join(os.getcwd(), "lyaloss_lqr.pth"),
        )

    if cfg.model.load_lyaloss is not None:
        load_lyaloss = cfg.model.load_lyaloss
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(2, dtype=dtype, device=device)
        )

    candidate_scale = cfg.loss.candidate_scale
    l1_reg = cfg.loss.l1_reg
    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa
    save_lyaloss_path = None

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )
        # wandb.config.update(cfg)

    limit_x = torch.tensor([0.75, 0.75, np.pi / 2, 4, 4, 3], dtype=dtype, device=device)
    if cfg.train.train_lyaloss:
        for n in range(len(cfg.model.limit_scale)):
            limit_scale = cfg.model.limit_scale[n]
            limit = limit_scale * limit_x
            lower_limit = -limit
            upper_limit = limit
            permute_array = [[-1, 1]] * quadrotor_continuous.nx
            permute_array_torch = torch.tensor(
                list(itertools.product(*permute_array)), device=device
            )

            derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
                dynamics,
                controller,
                lyapunov_nn,
                lower_limit,
                upper_limit,
                rho_multiplier,
                kappa=kappa,
                hard_max=hard_max,
                beta=1,
                loss_weights=torch.tensor([0.5, 1, 0.5], device=device),
            )
            candidate_roa_states = permute_array_torch * upper_limit
            if candidate_scale < 1:
                # Sample on level set of V_lqr and scale between (0, 1)
                V_candidate = V(candidate_roa_states)
                V_max = torch.max(V_candidate)
                candidate_roa_states = (
                    candidate_roa_states
                    / torch.sqrt(V_candidate / V_max)
                    * candidate_scale
                )
            else:
                x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                    lyapunov_nn,
                    lower_limit,
                    upper_limit,
                    num_samples_per_boundary=1000,
                    eps=limit,
                    steps=100,
                    direction="minimize",
                )
                # Sample slightly outside the current ROA
                V_candidate = lyapunov_nn(candidate_roa_states).clone().detach()
                candidate_roa_states = (
                    candidate_roa_states
                    / torch.sqrt(V_candidate / lyapunov_nn(x_min_boundary).min().item())
                    * candidate_scale
                )
            candidate_roa_states = torch.clamp(
                candidate_roa_states, min=lower_limit, max=upper_limit
            )

            save_name = f"lyaloss_{kappa}kappa_{limit_scale}.pth"
            if save_lyaloss:
                save_lyaloss_path = os.path.join(os.getcwd(), f"{save_name}")
            train_utils.train_lyapunov_with_buffer(
                derivative_lyaloss=derivative_lyaloss,
                positivity_lyaloss=positivity_lyaloss,
                observer_loss=None,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                grid_size=grid_size,
                learning_rate=cfg.train.learning_rate,
                lr_controller=cfg.train.lr_controller,
                weight_decay=0.0,
                max_iter=cfg.train.max_iter[n],
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
                l1_reg=cfg.loss.l1_reg[n],
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                V_decrease_within_roa=V_decrease_within_roa,
                Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
                Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
                candidate_roa_states=candidate_roa_states,
                candidate_roa_states_weight=cfg.loss.candidate_roa_states_weight[n],
                hard_max=hard_max,
                lr_scheduler=cfg.train.lr_scheduler,
                logger=logger,
            )

        torch.save(
            {
                "state_dict": lyapunov_nn.state_dict(),
                "rho": derivative_lyaloss.get_rho(),
            },
            os.path.join(os.getcwd(), "lyapunov_nn.pth"),
        )
    else:
        limit = cfg.model.limit_scale[-1] * limit_x
        lower_limit = -limit
        upper_limit = limit

    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        lower_limit,
        upper_limit,
        rho_multiplier=rho_multiplier,
        kappa=0e-3,
        hard_max=True,
    )
    # Check with pgd attack.
    for seed in range(50):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=1000,
                eps=limit,
                steps=cfg.pgd_verifier_steps,
                direction="minimize",
            )
            derivative_lyaloss_check.x_boundary = x_min_boundary
        x_check_start = (
            (
                torch.rand((50000, quadrotor_continuous.nx), dtype=dtype, device=device)
                - 0.5
            )
            * limit
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit,
            steps=200,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        logger.info(msg)
        x_adv = adv_x[(adv_lya < 0).squeeze()]
        logger.info(adv_lya.min().item())

    plt.clf()
    x0 = (
        (torch.rand((40, quadrotor_continuous.nx), dtype=dtype, device=device) - 0.5)
        * 2
        * limit
    )
    x_traj, V_traj = models.simulate(derivative_lyaloss, 800, x0)
    V_traj = torch.stack(V_traj).cpu().detach().squeeze().numpy()
    rho = derivative_lyaloss_check.get_rho().item()
    V_traj = V_traj[200:, V_traj[200, :] <= rho]
    plt.plot(dt * np.arange(V_traj.shape[0]), V_traj)
    plt.savefig(os.path.join(os.getcwd(), f"V_traj_{kappa}_{candidate_scale}.png"))

    print("rho = ", rho)
    labels = [r"$x$", r"$y$", r"$\theta$", r"$\dot x$", r"$\dot y$", r"$\dot \theta$"]
    for plot_idx in [[2, 5], [0, 2], [3, 4], [4, 5]]:
        fig = plt.figure()
        train_utils.plot_V_heatmap(
            fig,
            lyapunov_nn,
            rho,
            lower_limit,
            upper_limit,
            quadrotor_continuous.nx,
            x_boundary=derivative_lyaloss_check.x_boundary,
            plot_idx=plot_idx,
            mode=0.0,
            V_lqr=V,
        )
        # if plot_idx == [0, 2]:
        #     plt.xticks([-0.75, -0.75/2, 0, 0.75/2, 0.75], [r"$-0.75$", r"$-0.375$", r"$0$", r"$0.375$", r"$0.75$"], fontsize=15)
        #     plt.yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"], fontsize=15)
        # elif plot_idx == [3, 4]:
        #     plt.xticks([-4, -4/2, 0, 4/2, 4], [r"$-4$", r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize=15)
        #     plt.yticks([-4, -4/2, 0, 4/2, 4], [r"$-4$", r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize=15)
        # elif plot_idx == [4, 5]:
        #     plt.xticks([-4, -4/2, 0, 4/2, 4], [r"$-4$", r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize=15)
        #     plt.yticks([-3, -3/2, 0, 3/2, 3], [r"$-3$", r"$-1.5$", r"$0$", r"$1.5$", r"$3$"], fontsize=15)
        # elif plot_idx == [2, 5]:
        #     plt.xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"], fontsize=15)
        #     plt.yticks([-3, -3/2, 0, 3/2, 3], [r"$-3$", r"$-1.5$", r"$0$", r"$1.5$", r"$3$"], fontsize=15)

        # plt.title(f"rho = {rho}")
        plt.savefig(
            os.path.join(
                os.getcwd(), f"V_{kappa}_{candidate_scale}_roa_{str(plot_idx)}.png"
            )
        )


if __name__ == "__main__":
    main()
