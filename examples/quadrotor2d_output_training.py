import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import itertools

import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.output_train_utils as output_train_utils
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.quadrotor2d as quadrotor2d
import neural_lyapunov_training.train_utils as train_utils

import wandb
import os

device = torch.device("cuda")
dtype = torch.float


@hydra.main(config_path="./config", config_name="quadrotor2d_output_training")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    train_utils.set_seed(cfg.seed)

    quadrotor_continuous = quadrotor2d.Quadrotor2DLidarDynamics()
    dt = 0.01
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(quadrotor_continuous, dt)
    dynamics.to(device)

    nx = quadrotor_continuous.nx
    x_max = torch.tensor([1, np.pi / 2, 2, 2 * np.pi], device=device)
    e_max = x_max / 2
    limit_xe = torch.concat((x_max, e_max))
    limit_scale = cfg.model.limit_scale
    limit = limit_scale * limit_xe
    grid_size = torch.tensor([4, 6, 4, 8, 2, 3, 2, 3], device=device)
    lower_limit = -limit
    upper_limit = limit

    h = lambda x: quadrotor_continuous.h(x)

    x0 = (dynamics.x_equilibrium).to(device)
    controller = controllers.NeuralNetworkController(
        nlayer=2,
        in_dim=nx + quadrotor_continuous.ny,
        out_dim=2,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([0, 0.0], device=device),
        u_up=(dynamics.u_equilibrium * 3).to(device),
        x_equilibrium=torch.cat(
            (x0, torch.zeros(quadrotor_continuous.ny, device=device))
        ),
        u_equilibrium=(dynamics.u_equilibrium).to(device),
    )
    controller.to(device)
    controller.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/quadrotor2d/output_feedback/controller_[8, 8].pth",
            )
        )
    )
    controller.eval()

    # Reference EKF observer
    ekf_observer = controllers.EKFObserver(
        dynamics, h, gamma=0, delta=1e-3, lam=0, alpha=1.1
    )
    observer = controllers.NeuralNetworkLuenbergerObserver(
        nx,
        quadrotor_continuous.ny,
        dynamics,
        h,
        torch.zeros(1, quadrotor_continuous.ny),
        fc_hidden_dim=[8, 8],
    )
    observer.to(device)
    observer.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/quadrotor2d/output_feedback/observer_[8, 8].pth",
            )
        )
    )
    observer.eval()

    K, S = quadrotor_continuous.lqr_control()
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)

    V_lqr = lambda x: torch.sum(
        x * (x @ S_torch), axis=1, keepdim=True
    )  # Scale V_lqr to be in [0, 10]
    u_lqr = lambda x: x @ K_torch.T + quadrotor_continuous.u_equilibrium.to(device)
    controller_lqr = lambda x: torch.clamp(
        u_lqr(x),
        min=torch.tensor([0, 0.0], device=device),
        max=dynamics.u_equilibrium.to(device) * 3,
    )

    S_ratio = 25
    S_cl = S_ratio * torch.cat(
        (
            torch.cat((S_torch, torch.zeros(nx, nx, device=device)), dim=1),
            torch.cat(
                (
                    torch.zeros(nx, nx, device=device),
                    torch.linalg.inv(ekf_observer.P0.to(device)) / 50,
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    R = torch.linalg.cholesky(S_cl)
    lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
        goal_state=torch.zeros(8, dtype=dtype).to(device),
        x_dim=8,
        R_rows=8,
        eps=0.01,
        R=R,
    )
    lyapunov_nn.to(device)

    kappa = cfg.model.kappa
    hard_max = cfg.train.hard_max
    rho_multiplier = cfg.model.rho_multiplier
    derivative_lyaloss = lyapunov.LyapunovDerivativeDOFLoss(
        dynamics,
        observer,
        controller,
        lyapunov_nn,
        lower_limit,
        upper_limit,
        rho_multiplier,
        kappa=kappa,
        hard_max=hard_max,
        beta=1,
        loss_weights=torch.tensor([0.5, 1.0, 1.0], device=device),
    )
    observer_loss = lyapunov.ObserverLoss(dynamics, observer, controller, ekf_observer)

    if cfg.model.load_lyaloss is not None:
        load_lyaloss = cfg.model.load_lyaloss
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    candidate_scale = cfg.loss.candidate_scale
    l1_reg = cfg.loss.l1_reg
    candidate_roa_states_weight = cfg.loss.candidate_roa_states_weight

    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa
    save_lyaloss_path = None
    save_name = f"lyaloss_{kappa}kappa_{l1_reg}.pth"
    if save_lyaloss:
        save_lyaloss_path = os.path.join(os.getcwd(), f"{save_name}")

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )

    if cfg.train.train_lyaloss:
        permute_array = [[-1.0, 1.0]] * 2 * nx
        permute_array_torch = torch.tensor(
            list(itertools.product(*permute_array)), device=device
        )
        candidate_roa_states = permute_array_torch * upper_limit
        # Sample slightly outside the current ROA
        V_candidate = lyapunov_nn(candidate_roa_states).clone().detach()
        x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=1000,
            eps=limit,
            steps=100,
            direction="minimize",
        )
        candidate_roa_states = (
            candidate_roa_states
            / torch.sqrt(V_candidate / lyapunov_nn(x_min_boundary).min().item())
            * candidate_scale
        )
        candidate_roa_states = torch.clamp(
            candidate_roa_states, min=lower_limit, max=upper_limit
        )

        train_utils.train_lyapunov_with_buffer(
            derivative_lyaloss=derivative_lyaloss,
            positivity_lyaloss=None,
            observer_loss=observer_loss,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            grid_size=grid_size,
            learning_rate=cfg.train.learning_rate,
            lr_controller=cfg.train.lr_controller,
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
            observer_ratio=cfg.loss.observer_ratio,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            V_decrease_within_roa=V_decrease_within_roa,
            Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
            Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
            candidate_roa_states=candidate_roa_states,
            candidate_roa_states_weight=cfg.loss.candidate_roa_states_weight,
            hard_max=hard_max,
            lr_scheduler=cfg.train.lr_scheduler,
            always_candidate_roa_regulizer=cfg.loss.always_candidate_roa_regulizer,
        )

    # "Verify" Lyapunov conditions with PGD attack
    derivative_lyaloss_check = lyapunov.LyapunovDerivativeDOFLoss(
        dynamics,
        observer,
        controller,
        lyapunov_nn,
        lower_limit,
        upper_limit,
        rho_multiplier=rho_multiplier,
        kappa=0e-3,
        hard_max=True,
    )
    for seed in range(50):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=1000,
                eps=limit,
                steps=100,
                direction="minimize",
            )
            derivative_lyaloss_check.x_boundary = x_min_boundary
        x_check_start = (
            (
                torch.rand((10000, 2 * nx), device=device)
                - torch.full((2 * nx,), 0.5, device=device)
            )
            * limit
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit,
            steps=100,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        print(
            f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        )

    rho = derivative_lyaloss_check.get_rho().item()
    print("rho = ", rho)

    # Simulate the system
    plt.clf()
    x0 = (
        (torch.rand((40, quadrotor_continuous.nx), device=device) - 0.5)
        * 2
        * limit[:nx]
    )
    e0 = (
        (torch.rand((40, quadrotor_continuous.nx), device=device) - 0.5)
        * 2
        * limit[nx:]
    )
    z0 = x0 - e0
    x_traj, z_traj, V_traj = output_train_utils.simulate(
        derivative_lyaloss, 500, x0, z0
    )
    e_traj = x_traj - z_traj
    V_traj = V_traj[100:]
    idx = V_traj[0, :] <= rho
    V_traj = V_traj[:, idx]
    plt.plot(dt * np.arange(400), V_traj)
    plt.savefig(os.path.join(os.getcwd(), f"V_traj_{kappa}.png"))

    x_boundary = x_min_boundary[torch.argmin(lyapunov_nn(x_min_boundary))]
    labels = [
        r"$y$",
        r"$\theta$",
        r"$\dot y$",
        r"$\dot \theta$",
        r"$e_y$",
        r"$e_\theta$",
        r"$e_{\dot y}$",
        r"$e_{\dot \theta}$",
    ]
    for plot_idx in [[0, 1], [2, 3], [4, 5], [6, 7]]:
        fig = plt.figure()
        train_utils.plot_V_heatmap(
            fig,
            lyapunov_nn,
            rho,
            lower_limit,
            upper_limit,
            8,
            x_boundary,
            plot_idx=plot_idx,
            mode=0.0,
        )
        # plt.title(f"rho = {rho}")
        if plot_idx == [2, 3]:
            plt.xticks(
                [-0.2, -0.2 / 2, 0, 0.2 / 2, 0.2],
                [r"$-0.2$", r"$-0.1$", r"$0$", r"$0.1$", r"$0.2$"],
                fontsize=15,
            )
            plt.yticks(
                [-np.pi * 0.2, -np.pi * 0.1, 0, np.pi * 0.1, np.pi * 0.2],
                [r"$-0.2\pi$", r"$-0.1\pi$", r"$0$", r"$0.1\pi$", r"$0.2\pi$"],
                fontsize=15,
            )
        elif plot_idx == [4, 5]:
            plt.xticks(
                [-0.05, -0.025, 0, 0.025, 0.05],
                [r"$-0.05$", r"$-0.025$", r"$0$", r"$0.025$", r"$0.05$"],
                fontsize=15,
            )
            plt.yticks(
                [-np.pi * 0.025, -np.pi * 0.0125, 0, np.pi * 0.0125, np.pi * 0.025],
                [
                    r"$-0.025\pi$",
                    r"$-0.0125\pi$",
                    r"$0$",
                    r"$0.0125\pi$",
                    r"$0.025\pi$",
                ],
                fontsize=15,
            )
        elif plot_idx == [6, 7]:
            plt.xticks(
                [-0.1, -0.1 / 2, 0, 0.1 / 2, 0.1],
                [r"$-0.1$", r"$-0.05$", r"$0$", r"$0.05$", r"$0.1$"],
                fontsize=15,
            )
            plt.yticks(
                [-np.pi * 0.1, -np.pi * 0.05, 0, np.pi * 0.05, np.pi * 0.1],
                [r"$-0.1\pi$", r"$-0.05\pi$", r"$0$", r"$0.05\pi$", r"$0.1\pi$"],
                fontsize=15,
            )

        plt.savefig(
            os.path.join(
                os.getcwd(),
                f"V_{kappa}_{candidate_roa_states_weight}_{str(plot_idx)}.png",
            )
        )


if __name__ == "__main__":
    main()
