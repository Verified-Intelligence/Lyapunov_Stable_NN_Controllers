import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.output_train_utils as output_train_utils
import itertools
import pendulum_state_training as pt

import neural_lyapunov_training.train_utils as train_utils
import wandb
import os

device = torch.device("cuda")
dtype = torch.float


@hydra.main(config_path="./config", config_name="pendulum_output_training")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    train_utils.set_seed(cfg.seed)

    grid_size = torch.tensor([10, 10, 5, 5], device=device)

    dt = 0.01
    pendulum_continuous = pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        pendulum_continuous, dt=dt
    )
    nx = pendulum_continuous.nx
    dynamics.nx = 4
    dynamics.to(device)

    # Reference controller
    u_max = cfg.model.u_max  # mgl = 0.736
    u_nominal = output_train_utils.load_sos_controller(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data/pendulum/output_feedback/sos_controller.pkl",
        ),
        lambda x: x,
        2,
    )
    controller_nominal = lambda x: torch.clamp(u_nominal(x), min=-u_max, max=u_max)

    controller = controllers.NeuralNetworkController(
        nlayer=4,
        in_dim=3,
        out_dim=1,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([-u_max]),
        u_up=torch.tensor([u_max]),
        x_equilibrium=torch.zeros(3, dtype=dtype),
        u_equilibrium=pendulum_continuous.u_equilibrium,
    )
    controller.to(device)
    controller.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/pendulum/output_feedback/controller_[8, 8, 8].pth",
            )
        )
    )
    controller.eval()

    h = lambda x: pendulum_continuous.h(x)
    # Reference EKF observer
    ekf_observer = controllers.EKFObserver(
        dynamics, h, gamma=0, delta=1e-3, lam=0.1, alpha=1.05
    )
    observer = controllers.NeuralNetworkLuenbergerObserver(
        nx,
        pendulum_continuous.ny,
        dynamics,
        h,
        torch.zeros(1, pendulum_continuous.ny),
        fc_hidden_dim=[8, 8],
    )
    observer.to(device)
    observer.load_state_dict(
        torch.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/pendulum/output_feedback/observer_[8, 8].pth",
            )
        )
    )
    observer.eval()

    K, S = pt.compute_lqr(pendulum_continuous)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    V_lqr = lambda x: torch.sum(x * (x @ S_torch), axis=1, keepdim=True)

    if cfg.model.lyapunov.quadratic:
        S_cl = torch.cat(
            (
                torch.cat((S_torch / 10, torch.zeros(nx, nx, device=device)), dim=1),
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
            goal_state=torch.zeros(4, dtype=dtype).to(device),
            x_dim=4,
            R_rows=4,
            eps=0.01,
            R=R,
        )

    else:
        lyapunov_hidden_widths = cfg.model.lyapunov.hidden_widths
        lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
            goal_state=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            hidden_widths=lyapunov_hidden_widths,
            x_dim=4,
            R_rows=2,
            absolute_output=True,
            eps=0.01,
            activation=nn.LeakyReLU,
            V_psd_form=cfg.model.V_psd_form,
        )
        lyapunov_nn.load_state_dict(
            torch.load(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "data/pendulum/output_feedback/lyapunov_init.pth",
                )
            )
        )
        lyapunov_nn.eval()
    lyapunov_nn.to(device)

    def lyapunov_target(xe, r=0.01):
        x = xe[:, :nx]
        e = xe[:, nx:]
        Vc = V_lqr(x)
        Vo = (
            torch.einsum(
                "bi, ii, bi->b", e, torch.linalg.inv(ekf_observer.P0.to(e.device)), e
            )
            / 200
        )
        return r * Vc + Vo.unsqueeze(1)

    kappa = cfg.model.kappa
    hard_max = cfg.train.hard_max
    # Placeholder for loading models
    derivative_lyaloss = lyapunov.LyapunovDerivativeDOFLoss(
        dynamics,
        observer,
        controller,
        lyapunov_nn,
        0,
        0,
        1,
        kappa=kappa,
        hard_max=hard_max,
        beta=1,
        loss_weights=torch.tensor([0.5, 1.0, 0.5], device=device),
    )
    observer_loss = lyapunov.ObserverLoss(dynamics, observer, controller, ekf_observer)

    if cfg.model.load_lyaloss is not None:
        load_lyaloss = cfg.model.load_lyaloss
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    positivity_lyaloss = None

    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa
    save_lyaloss_path = None

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )

    limit_xe = torch.tensor([np.pi, np.pi, np.pi / 4, np.pi / 4], device=device)
    if cfg.train.train_lyaloss:
        permute_array = [[-1.0, 1.0]] * 4
        permute_array_torch = torch.tensor(
            list(itertools.product(*permute_array)), device=device
        )

        for n in range(len(cfg.model.limit_scale)):
            limit_scale = cfg.model.limit_scale[n]
            limit = limit_xe * limit_scale
            lower_limit = -limit
            upper_limit = limit
            candidate_roa_states = permute_array_torch * upper_limit
            V_candidate = lyapunov_target(candidate_roa_states)
            V_max = torch.max(V_candidate)
            candidate_roa_states = (
                candidate_roa_states
                / torch.sqrt(V_candidate / V_max)
                * cfg.loss.candidate_scale
            )

            rho_multiplier = cfg.model.rho_multiplier[n]
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
                loss_weights=torch.tensor([0.5, 1.0, 0.5], device=device),
            )

            save_name = f"lyaloss_{kappa}kappa_{limit_scale}_{u_max}.pth"
            if save_lyaloss:
                save_lyaloss_path = os.path.join(os.getcwd(), f"{save_name}")

            train_utils.train_lyapunov_with_buffer(
                derivative_lyaloss=derivative_lyaloss,
                positivity_lyaloss=positivity_lyaloss,
                observer_loss=observer_loss,
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
                observer_ratio=cfg.loss.observer_ratio[n],
                Vmin_x_pgd_buffer_size=cfg.train.Vmin_x_pgd_buffer_size,
                V_decrease_within_roa=V_decrease_within_roa,
                Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
                Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
                candidate_roa_states=candidate_roa_states,
                candidate_roa_states_weight=cfg.loss.candidate_roa_states_weight[n],
                lr_scheduler=cfg.train.lr_scheduler,
                hard_max=cfg.train.hard_max,
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
        limit = limit_xe * cfg.model.limit_scale[-1]
        lower_limit = -limit
        upper_limit = limit
        rho_multiplier = cfg.model.rho_multiplier[-1]

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
                torch.rand((50000, 4), device=device)
                - torch.full((4,), 0.5, device=device)
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
        print(
            f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        )

    rho = derivative_lyaloss_check.get_rho().item()
    print("rho = ", rho)

    # Simulate the system
    plt.clf()
    x_max = limit[:2]
    e_max = limit[2:]
    x0 = (torch.rand((50, 2), device=device) - 0.5) * 2 * x_max
    e0 = (torch.rand((50, 2), device=device) - 0.5) * 2 * e_max

    # n_grid = 7
    # X1, X2 = torch.meshgrid(torch.linspace(x_min[0], x_max[0], n_grid, device=device),
    #                 torch.linspace(x_min[1], x_max[1], n_grid, device=device))
    # x0 = torch.vstack((X1.flatten(), X2.flatten())).transpose(0, 1)
    # e0 = (torch.rand((x0.shape[0], 2), device=device) - 0.5) * 2 * e_max

    z0 = x0 - e0
    x_traj, z_traj, V_traj = output_train_utils.simulate(
        derivative_lyaloss_check, 1400, x0, z0
    )
    e_traj = x_traj - z_traj
    idx = V_traj[100, :] <= rho
    V_traj = V_traj[:, idx]
    plt.plot(dt * np.arange(V_traj.shape[0]), V_traj)
    plt.title(f"{max_adv_violation}, {adv_output.sum().item()}")
    plt.savefig(os.path.join(os.getcwd(), f"V_traj_{kappa}.png"))

    x_boundary = x_min_boundary[torch.argmin(lyapunov_nn(x_min_boundary))]
    labels = [r"$\theta$", r"$\dot \theta$", r"$e_\theta$", r"$e_{\dot \theta}$"]
    for plot_idx in [[0, 1], [2, 3]]:
        fig = plt.figure()
        train_utils.plot_V_heatmap(
            fig,
            lyapunov_nn,
            rho,
            lower_limit,
            upper_limit,
            2 * pendulum_continuous.nx,
            x_boundary,
            plot_idx=plot_idx,
            mode=0.0,
        )
        # plt.title(f"rho = {rho}")
        if plot_idx[0] == 0:
            # plt.xticks(
            #     [-0.7 * np.pi, -0.7 / 2 * np.pi, 0, 0.7 / 2 * np.pi, 0.7 * np.pi],
            #     [rf"$-0.7\pi$", rf"$-0.35\pi$", r"$0$", rf"$0.35\pi$", rf"$0.7\pi$"],
            #     fontsize=15,
            # )
            # plt.yticks(
            #     [-0.7 * np.pi, -0.7 / 2 * np.pi, 0, 0.7 / 2 * np.pi, 0.7 * np.pi],
            #     [rf"$-0.7\pi$", rf"$-0.35\pi$", r"$0$", rf"$0.35\pi$", rf"$0.7\pi$"],
            #     fontsize=15,
            # )

            # plt.xticks(
            #     [-np.pi, -1 / 2 * np.pi, 0, 1 / 2 * np.pi, np.pi],
            #     [rf"$-\pi$", rf"$-0.5\pi$", r"$0$", rf"$0.5\pi$", rf"$\pi$"],
            #     fontsize=15,
            # )
            # plt.yticks(
            #     [-np.pi, - 1/ 2 * np.pi, 0,  1/ 2 * np.pi, np.pi],
            #     [rf"$-\pi$", rf"$-0.5\pi$", r"$0$", rf"$0.5\pi$", rf"$\pi$"],
            #     fontsize=15,
            # )

            # plt.xticks(
            #     [-0.4 * np.pi, -0.4 / 2 * np.pi, 0, 0.4 / 2 * np.pi, 0.4 * np.pi],
            #     [rf"$-0.4\pi$", rf"$-0.2\pi$", r"$0$", rf"$0.2\pi$", rf"$0.4\pi$"],
            #     fontsize=15,
            # )
            # plt.yticks(
            #     [-0.4 * np.pi, -0.4 / 2 * np.pi, 0, 0.4 / 2 * np.pi, 0.4 * np.pi],
            #     [rf"$-0.4\pi$", rf"$-0.2\pi$", r"$0$", rf"$0.2\pi$", rf"$0.4\pi$"],
            #     fontsize=15,
            # )

            plt.plot(x_traj[:, :, 0], x_traj[:, :, 1], linewidth=2)
        else:
            # plt.xticks(
            #     [-0.14 * np.pi, -0.07 * np.pi, 0, 0.07 * np.pi, 0.14 * np.pi],
            #     [rf"$-0.14\pi$", rf"$-0.07\pi$", r"$0$", rf"$0.07\pi$", rf"$0.14\pi$"],
            #     fontsize=15,
            # )
            # plt.yticks(
            #     [-0.14 * np.pi, -0.07 * np.pi, 0, 0.07 * np.pi, 0.14 * np.pi],
            #     [rf"$-0.14\pi$", rf"$-0.07\pi$", r"$0$", rf"$0.07\pi$", rf"$0.14\pi$"],
            #     fontsize=15,
            # )

            # plt.xticks(
            #     [-0.1 * np.pi, -0.05 * np.pi, 0, 0.05 * np.pi, 0.1 * np.pi],
            #     [rf"$-0.1\pi$", rf"$-0.05\pi$", r"$0$", rf"$0.05\pi$", rf"$0.1\pi$"],
            #     fontsize=15,
            # )
            # plt.yticks(
            #     [-0.1 * np.pi, -0.05 * np.pi, 0, 0.05 * np.pi, 0.1 * np.pi],
            #     [rf"$-0.1\pi$", rf"$-0.05\pi$", r"$0$", rf"$0.05\pi$", rf"$0.1\pi$"],
            #     fontsize=15,
            # )

            # plt.xticks(
            #     [-0.25 * np.pi, -0.125 * np.pi, 0, 0.125 * np.pi, 0.25 * np.pi],
            #     [r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$"],
            #     fontsize=15,
            # )
            # plt.yticks(
            #     [-0.25 * np.pi, -0.125 * np.pi, 0, 0.125 * np.pi, 0.25 * np.pi],
            #     [r"$-\frac{\pi}{4}$", r"$-\frac{\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$"],
            #     fontsize=15,
            # )
            plt.plot(e_traj[:, :, 0], e_traj[:, :, 1], linewidth=2)
        plt.savefig(os.path.join(os.getcwd(), f"V_{kappa}_{str(plot_idx)}.png"))


if __name__ == "__main__":
    main()
