import os

import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import scipy
import torch
import torch.nn as nn
import wandb

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.path_tracking as path_tracking
import neural_lyapunov_training.train_utils as train_utils

device = torch.device("cuda")
dtype = torch.float


def compute_lqr(path_tracking_continuous: path_tracking.PathTrackingDynamics):
    x_equilibrium = path_tracking_continuous.x_equilibrium.to(device)
    u_equilibrium = path_tracking_continuous.u_equilibrium.to(device)
    A_batch, B_batch = path_tracking_continuous.linearized_dynamics(
        x_equilibrium.unsqueeze(0), u_equilibrium.unsqueeze(0)
    )
    A = A_batch.squeeze(0).cpu().detach().numpy()
    B = B_batch.squeeze(0).cpu().detach().numpy()
    Q = np.eye(2)
    R = np.eye(1)
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, B.T @ S)
    return K, S


def approximate_lqr(
    path_tracking_continuous: path_tracking.PathTrackingDynamics,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn: lyapunov.NeuralNetworkLyapunov,
    upper_limit: torch.Tensor,
    logger,
):
    K, S = compute_lqr(path_tracking_continuous)
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
    ax.set_xlabel("distance")
    ax.set_ylabel("angle")
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar


@hydra.main(config_path="./config", config_name="path_tracking_state_training.yaml")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    train_utils.set_seed(cfg.seed)

    dt = cfg.model.dt
    path_tracking_continuous = path_tracking.PathTrackingDynamics(
        speed=2.0, length=1.0, radius=10.0
    )
    dynamics = dynamical_system.FirstOrderDiscreteTimeSystem(
        path_tracking_continuous,
        dt=dt,
        integration=dynamical_system.IntegrationMethod[cfg.model.integration],
    )

    controller = controllers.NeuralNetworkController(
        nlayer=4,
        in_dim=2,
        out_dim=1,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([-0.84]),
        u_up=torch.tensor([0.84]),
        x_equilibrium=path_tracking_continuous.x_equilibrium,
        u_equilibrium=path_tracking_continuous.u_equilibrium,
    )
    controller.eval()

    absolute_output = True
    if cfg.model.lyapunov.quadratic:
        _, S = compute_lqr(path_tracking_continuous)
        S_torch = torch.from_numpy(S).type(dtype).to(device)
        R = torch.linalg.cholesky(S_torch)
        lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
            goal_state=torch.zeros(2, dtype=dtype).to(device),
            x_dim=2,
            R_rows=2,
            eps=0.01,
            R=R,
        )
    else:
        lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
            goal_state=path_tracking_continuous.x_equilibrium,
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
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=0,
        box_up=0,
        rho_multiplier=1,
        kappa=kappa,
        hard_max=cfg.train.hard_max,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    grid_size = torch.tensor([50, 50], device=device)
    logger = logging.getLogger(__name__)
    if cfg.approximate_lqr:
        approximate_lqr(
            path_tracking_continuous, controller, lyapunov_nn, upper_limit, logger
        )
        torch.save(
            {"state_dict": derivative_lyaloss.state_dict()},
            os.path.join(os.getcwd(), "lyaloss_lqr.pth"),
        )

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

    if cfg.train.derivative_x_buffer_path is not None:
        derivative_x_buffer = torch.load(cfg.train.derivative_x_buffer_path)
    else:
        derivative_x_buffer = None

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
                rho_multiplier=cfg.model.rho_multiplier[n],
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
                derivative_x_buffer=derivative_x_buffer,
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

    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=cfg.model.rho_multiplier[-1],
        kappa=0.0,
        hard_max=True,
    )
    pgd_verifier_find_counterexamples = False
    counterexamples_check = torch.zeros((0, 2), device=device)
    for seed in range(100):
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
        counterexamples_check = torch.cat(
            (counterexamples_check, adv_x[adv_output.squeeze(1) > 0]), dim=0
        )
        if max_adv_violation > 0:
            pgd_verifier_find_counterexamples = True
        logger.info(msg)

    logger.info(
        f"PGD verifier finds counter examples? {pgd_verifier_find_counterexamples}"
    )
    if counterexamples_check.shape[0] > 0:
        torch.save(
            counterexamples_check,
            os.path.join(os.getcwd(), "counterexamples_check.pth"),
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
        path_tracking_continuous.nx,
        derivative_lyaloss.x_boundary,
    )
    fig.show()
    plt.savefig(os.path.join(os.getcwd(), "V_roa.png"))


if __name__ == "__main__":
    main()
