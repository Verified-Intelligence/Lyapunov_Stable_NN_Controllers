import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
import itertools

import neural_lyapunov_training.arguments as arguments
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.pvtol as pvtol
import neural_lyapunov_training.train_utils as train_utils
import neural_lyapunov_training.output_train_utils as output_train_utils

device = torch.device("cuda")
dtype = torch.float


# def approximate_lqr(pvtol,
#                     controller: controllers.NeuralNetworkController,
#                     lyapunov_nn: lyapunov.NeuralNetworkLyapunov):

#     Q = np.diag(np.array([1, 1, 1, 10, 10, 10.]))
#     R = np.diag(np.array([10, 10.]))
#     K, S = pvtol.lqr_control(Q, R, pvtol.x_equilibrium,
#                                  pvtol.u_equilibrium)
#     K_torch = torch.from_numpy(K).type(dtype).to(device)
#     S_torch = torch.from_numpy(S).type(dtype).to(device)

#     # We will sample x during each training iteration.
#     V = lambda x: torch.sum(x * (x @ S_torch), axis=1, keepdim=True)/50
#     u = lambda x: x @ K_torch.T + pvtol.u_equilibrium.to(device)

#     def approximate(system, target_func, lr, max_iter, l1_reg=1.0):
#         optimizer = torch.optim.Adam(system.parameters(), lr=lr)
#         scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
#                                                       start_factor=1.0,
#                                                       end_factor=0.0,
#                                                       total_iters=max_iter)
#         total_elements = sum(p.numel() for p in system.parameters())
#         for i in range(max_iter):
#             optimizer.zero_grad(set_to_none=True)
#             # Sample x and compute target.
#             x = (torch.rand((100000, 6), device=device) - 0.5) * limit
#             y = target_func(x)
#             output = torch.nn.MSELoss()(system(x), y)
#             # Compute a L1 norm loss to encourage tighter IBP bounds.
#             l1_loss = l1_reg * sum(
#                 p.abs().sum() for p in system.parameters()) / total_elements
#             loss = output + l1_loss
#             loss.backward()
#             print(
#                 f"iter {i}, mse {output.item()}, l1 {l1_loss.item()}, loss {loss.item()}, lr {scheduler.get_last_lr()[0]:.5f}"
#             )
#             optimizer.step()
#             scheduler.step()

#     # TODO: tune L1 reg term.
#     approximate(lyapunov_nn, V, lr=0.02, max_iter=5000, l1_reg=0.01)
#     if len(
#             list(controller.parameters())
#     ) > 0:  # Do not train if there are no parameters (e.g., fixed linear controller).
#         approximate(controller, u, lr=0.05, max_iter=500, l1_reg=0.01)


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


def plot_V_heatmap(
    V,
    lower_limit,
    upper_limit,
    nx,
    labels,
    x_boundary=None,
    plot_idx=[0, 2],
    mode="boundary",
    V_lqr=None,
):
    x_ticks = torch.linspace(
        lower_limit[plot_idx[0]], upper_limit[plot_idx[0]], 50, device=device
    )
    y_ticks = torch.linspace(
        lower_limit[plot_idx[1]], upper_limit[plot_idx[1]], 50, device=device
    )
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    if mode == "boundary":
        X = torch.ones(2500, nx, device=device) * x_boundary
    elif isinstance(mode, float):
        X = torch.ones(2500, nx, device=device) * upper_limit * mode
    X[:, plot_idx[0]] = grid_x.flatten()
    X[:, plot_idx[1]] = grid_y.flatten()

    with torch.no_grad():
        V_val = V(X)

    V_val = V_val.cpu().reshape(50, 50)
    grid_x = grid_x.cpu()
    grid_y = grid_y.cpu()

    # Compute min V(x) on the boundary
    x_pgd_boundary_min = train_utils.calc_V_extreme_on_boundary_pgd(
        V,
        lower_limit,
        upper_limit,
        num_samples_per_boundary=1000,
        eps=(upper_limit - lower_limit) / 2,
        steps=100,
        direction="minimize",
    )
    rho_roa = torch.min(V(x_pgd_boundary_min)).item()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x, grid_y, V_val)
    ax.contour(grid_x, grid_y, V_val, [rho_roa], colors="red")
    if V_lqr is not None:
        V_lqr_val = V_lqr(X).reshape(50, 50).cpu()
        x_pgd_boundary_min = train_utils.calc_V_extreme_on_boundary_pgd(
            V_lqr,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=1000,
            eps=(upper_limit - lower_limit) / 2,
            steps=100,
            direction="minimize",
        )
        rho_lqr_roa = torch.min(V_lqr(x_pgd_boundary_min)).item()
        ax.contour(grid_x, grid_y, V_lqr_val, [rho_lqr_roa], colors="cyan")
    lower_limit = lower_limit.cpu()
    upper_limit = upper_limit.cpu()
    ax.set_xlim(lower_limit[plot_idx[0]], upper_limit[plot_idx[0]])
    ax.set_ylim(lower_limit[plot_idx[1]], upper_limit[plot_idx[1]])
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar


if __name__ == "__main__":
    # arguments.Config.add_argument(
    #     "--approximate_lqr_lyaloss",
    #     type=str,
    #     default=None,
    #     help=
    #     "Approximate LQR controller/cost and store the approximation model to this path.",
    #     hierarchy=["train", "approximate_lqr_lyaloss"])
    arguments.Config.add_argument(
        "--hard_max",
        type=bool,
        default=True,
        help="Softmax or hard max for the max in derivative lyapunov loss and candidate roa regularizer.",
        hierarchy=["train", "hard_max"],
    )
    arguments.Config.add_argument(
        "--candidate_scale",
        type=float,
        default=2.0,
        help="Scaling of al the vertices of the bounding box to be state-of-interest.",
        hierarchy=["loss", "candidate_scale"],
    )
    arguments.Config.add_argument(
        "--num_samples_per_boundary",
        type=int,
        default=500,
        help="Number of samples on the boundary for obtaining V_min using pgd attack.",
        hierarchy=["train", "num_samples_per_boundary"],
    )

    arguments.Config.parse_config()
    train_utils.set_seed(arguments.Config["general"]["seed"])

    if arguments.Config["general"]["dump_path"]:
        arguments.Config.dump_config(
            arguments.Config.all_args,
            out_to_doc=arguments.Config["general"]["dump_path"],
        )

    dt = 0.05
    pvtol_continuous = dynamics = pvtol.PvtolDynamics(dt=dt)

    print("Check equilibrium")
    print(
        dynamics(
            pvtol_continuous.x_equilibrium.unsqueeze(0),
            pvtol_continuous.u_equilibrium.unsqueeze(0),
        )
    )

    limit_scale = arguments.Config["model"]["limit_scale"]
    # limit = limit_scale * torch.tensor([0.75, 0.75, np.pi/2, 4, 4, 3],
    #                                     dtype=dtype, device=device)
    limit = limit_scale * torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device
    )
    lower_limit = -limit
    upper_limit = limit
    grid_size = torch.tensor([4, 4, 6, 5, 5, 6], device=device)

    Q = np.diag(np.array([1, 1, 1, 10, 10, 10.0]))
    R = np.diag(np.array([10, 10.0]))
    K, S = pvtol_continuous.lqr_control(
        Q, R, pvtol_continuous.x_equilibrium, pvtol_continuous.u_equilibrium
    )
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)

    V = (
        lambda x: torch.sum(x * (x @ S_torch), axis=1, keepdim=True) / 50
    )  # Scale V_lqr to be in [0, 10]
    u = lambda x: x @ K_torch.T + pvtol_continuous.u_equilibrium.to(device)

    # From Junlin Wu's code
    # lqr_weight = nn.Parameter(torch.FloatTensor(
    #     [[0.70710678, -0.70710678, -5.03954871,  1.10781077, -1.82439774, -1.20727555],
    #     [-0.70710678, -0.70710678,  5.03954871, -1.10781077, -1.82439774, 1.20727555]]))
    # controller_target = lambda x: x.matmul(lqr_weight.t())

    max_u = torch.tensor([1, 1.0], device=device) * 39.2
    controller_target = lambda x: torch.clamp(
        u(x), min=torch.tensor([0, 0.0], device=device), max=max_u
    )

    controller = controllers.NeuralNetworkController(
        nlayer=2,
        in_dim=6,
        out_dim=2,
        hidden_dim=8,
        clip_output="clamp",
        u_lo=torch.tensor([0, 0.0], dtype=dtype, device=device),
        u_up=max_u,
        x_equilibrium=(dynamics.x_equilibrium).to(device).to(dtype),
        u_equilibrium=(dynamics.u_equilibrium).to(device).to(dtype),
    )

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

    output_train_utils.approximate_controller(
        controller_target,
        controller,
        6,
        limit,
        0,
        lambda x: pvtol_continuous.h(x),
        "examples/data/pvtol/controller_lqr_[8, 8].pth",
        max_iter=500,
        lr=0.08,
        l1_reg=0.001,
    )
    # output_train_utils.approximate_controller(V, lyapunov_nn, 6, limit, 0, 0, "examples/data/pvtol/lyapunov_{}.pth".format(lyapunov_hidden_widths), max_iter=500, lr=0.05, l1_reg=0.01)

    controller.load_state_dict(
        torch.load("examples/data/pvtol/controller_lqr_[8, 8].pth")
    )
    controller.eval()

    kappa = arguments.Config["model"]["kappa"]
    hard_max = arguments.Config["train"]["hard_max"]
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics, controller, lyapunov_nn, kappa=kappa, hard_max=hard_max
    )

    # if arguments.Config["train"]["approximate_lqr_lyaloss"] is not None:
    #     approximate_lqr(pvtol_continuous, controller, lyapunov_nn)
    #     torch.save({"state_dict": derivative_lyaloss.state_dict()},
    #                arguments.Config["train"]["approximate_lqr_lyaloss"])

    if arguments.Config["model"]["load_lyaloss"] is not None:
        load_lyaloss = arguments.Config["model"]["load_lyaloss"]
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(2, dtype=dtype, device=device)
        )

    candidate_scale = arguments.Config["loss"]["candidate_scale"]
    candidate_roa_states_weight = arguments.Config["loss"][
        "candidate_roa_states_weight"
    ]
    data_folder = f"examples/data/pvtol/{limit_scale}"
    os.makedirs(data_folder, exist_ok=True)
    save_lyaloss = arguments.Config["model"]["save_lyaloss"]
    V_decrease_within_roa = arguments.Config["model"]["V_decrease_within_roa"]
    save_lyaloss_path = None
    save_name = (
        f"lyaloss_{kappa}kappa_{candidate_scale}_{candidate_roa_states_weight}.pth"
    )
    if save_lyaloss:
        save_lyaloss_path = f"{data_folder}/{save_name}"

    if arguments.Config["train"]["enable_wandb"]:
        wandb.init(project="pvtol", entity="zshi")
        wandb.config.update(arguments.Config.all_args)
        wandb.run.name = f"{limit_scale}/{save_name}"

    if arguments.Config["train"]["train_lyaloss"]:
        permute_array = [[-1, 1]] * pvtol_continuous.nx
        permute_array_torch = torch.tensor(
            list(itertools.product(*permute_array)), device=device
        )
        candidate_roa_states = permute_array_torch * upper_limit
        if candidate_scale < 1:
            # Sample on level set of V_lqr and scale between (0, 1)
            V_candidate = V(candidate_roa_states)
            V_max = torch.max(V_candidate)
            candidate_roa_states = (
                candidate_roa_states / torch.sqrt(V_candidate / V_max) * candidate_scale
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
            rho = lyapunov_nn(x_min_boundary).min().item()
            # Sample slightly outside the current ROA
            V_candidate = lyapunov_nn(candidate_roa_states).clone().detach()
            candidate_roa_states = (
                candidate_roa_states / torch.sqrt(V_candidate / rho) * candidate_scale
            )
        candidate_roa_states = torch.clamp(
            candidate_roa_states, min=lower_limit, max=upper_limit
        )
        train_utils.train_lyapunov_with_buffer(
            derivative_lyaloss=derivative_lyaloss,
            positivity_lyaloss=positivity_lyaloss,
            observer_loss=None,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            grid_size=grid_size,
            learning_rate=arguments.Config["train"]["learning_rate"],
            weight_decay=0.0,
            max_iter=arguments.Config["train"]["max_iter"],
            enable_wandb=arguments.Config["train"]["enable_wandb"],
            derivative_ibp_ratio=arguments.Config["loss"]["ibp_ratio_derivative"],
            derivative_sample_ratio=arguments.Config["loss"]["sample_ratio_derivative"],
            positivity_ibp_ratio=arguments.Config["loss"]["ibp_ratio_positivity"],
            positivity_sample_ratio=arguments.Config["loss"]["sample_ratio_positivity"],
            save_best_model=save_lyaloss_path,
            pgd_steps=arguments.Config["train"]["pgd_steps"],
            buffer_size=arguments.Config["train"]["buffer_size"],
            batch_size=arguments.Config["train"]["batch_size"],
            epochs=arguments.Config["train"]["epochs"],
            samples_per_iter=arguments.Config["train"]["samples_per_iter"],
            l1_reg=arguments.Config["loss"]["l1_reg"],
            num_samples_per_boundary=arguments.Config["train"][
                "num_samples_per_boundary"
            ],
            V_decrease_within_roa=V_decrease_within_roa,
            Vmin_x_boundary_weight=arguments.Config["loss"]["Vmin_x_boundary_weight"],
            Vmax_x_boundary_weight=arguments.Config["loss"]["Vmax_x_boundary_weight"],
            candidate_roa_states=candidate_roa_states,
            candidate_roa_states_weight=arguments.Config["loss"][
                "candidate_roa_states_weight"
            ],
            hard_max=hard_max,
            lr_scheduler=arguments.Config["train"]["lr_scheduler"],
        )

    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics, controller, lyapunov_nn, kappa=0e-3
    )
    fig, ax = plt.subplots(1, 2)
    # Check with pgd attack.
    for seed in range(200, 300):
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
            (torch.rand((50000, pvtol_continuous.nx), dtype=dtype, device=device) - 0.5)
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
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        print(msg)
        x_adv = adv_x[(adv_lya < 0).squeeze()]
        print(adv_lya.min().item())

    plt.clf()
    rho = lyapunov_nn(x_min_boundary).min().item()
    x0 = (
        (torch.rand((40, pvtol_continuous.nx), dtype=dtype, device=device) - 0.5)
        * 2
        * limit
    )
    x_traj, V_traj = models.simulate(derivative_lyaloss, 500, x0)
    V_traj = torch.stack(V_traj[100:]).cpu().detach().squeeze().numpy()
    V_traj = V_traj[:, V_traj[0, :] <= rho]
    plt.plot(dt * np.arange(400), V_traj)
    plt.savefig(f"{data_folder}/V_traj_{kappa}_{candidate_scale}.png")

    print("rho = ", rho)
    x_boundary = x_min_boundary[torch.argmin(lyapunov_nn(x_min_boundary))]
    print("Boundary state ratio = ", x_boundary / limit)
    labels = [r"$x$", r"$y$", r"$\theta$", r"$\dot x$", r"$\dot y$", r"$\dot \theta$"]
    for plot_idx in [[0, 1], [0, 2], [3, 4], [4, 5]]:
        fig2, axis2, cbar2 = plot_V_heatmap(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            6,
            labels,
            x_boundary,
            plot_idx=plot_idx,
            mode=0.0,
        )
        # plt.xticks([-0.75, -0.75/2, 0, 0.75/2, 0.75], [r"$-0.75$", r"$-0.375$", r"$0$", r"$0.375$", r"$0.75$"], fontsize=15)
        # plt.yticks([-0.75, -0.75/2, 0, 0.75/2, 0.75], [r"$-0.75$", r"$-0.375$", r"$0$", r"$0.375$", r"$0.75$"], fontsize=15)

        # plt.xticks([-4, -4/2, 0, 4/2, 4], [r"$-4$", r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize=15)
        # plt.yticks([-4, -4/2, 0, 4/2, 4], [r"$-4$", r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize=15)

        # plt.xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"], fontsize=15)
        # plt.yticks([-3, -3/2, 0, 3/2, 3], [r"$-3$", r"$-1.5$", r"$0$", r"$1.5$", r"$3$"], fontsize=15)

        # plt.title(f"rho = {rho}")
        plt.savefig(
            f"{data_folder}/V_{kappa}_{candidate_scale}_roa_{str(plot_idx)}.png"
        )
