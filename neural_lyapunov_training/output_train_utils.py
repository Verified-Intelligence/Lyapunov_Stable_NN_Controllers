import torch
import torch.nn as nn
import pickle
import numpy as np

import neural_lyapunov_training.lyapunov as lyapunov

device = torch.device("cuda")
dtype = torch.float


def load_sos_controller(input_file, x2z, nz):
    with open(input_file, "rb") as input_file:
        u_coeff = pickle.load(input_file)

    # Suboptimal state feedback controller synthesized from SOS
    def u(x):
        # notice that z = [s, c, theta_dot]
        poly = 0
        z = x2z(x)
        assert z.shape[1] == nz
        for monomial in u_coeff.keys():
            m = 1
            for i in range(nz):
                m *= z[:, i : i + 1].pow(monomial[i])
            poly += m * u_coeff[monomial]
        return poly

    return u


def train_q_extractor(
    extracter, h, nx, x_max, file_name, batch_size=100, lr=1e-3, max_iter=500, l1_reg=1
):
    print("Training extracter from observation to q")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(extracter.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)
    total_elements = sum(p.numel() for p in extracter.parameters())
    best_loss = np.inf

    for i in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        x = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * x_max
        y = h(x)
        x_pred = extracter(y)
        output = criterion(x[:, : int(nx / 2)], x_pred.squeeze())
        # Compute a L1 norm loss to encourage tighter IBP bounds.
        l1_loss = (
            l1_reg * sum(p.abs().sum() for p in extracter.parameters()) / total_elements
        )
        loss = output
        loss.backward()
        print(
            f"iter {i}, mse {output.item()}, l1 {l1_loss.item()}, loss {loss.item()}, lr {optimizer.param_groups[0]['lr']}"
        )
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(extracter.state_dict(), file_name)


def approximate_controller(
    controller_target,
    controller_nn,
    nx,
    x_max,
    e_max,
    h,
    file_name,
    batch_size=1000,
    lr=5e-3,
    max_iter=100,
    l1_reg=0.1,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(controller_nn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)
    total_elements = sum(p.numel() for p in controller_nn.parameters())
    best_loss = np.inf

    for i in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        x = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * x_max
        e = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * e_max
        z = x - e
        ey = h(x) - h(z)
        u_target = controller_target(x)
        u_pred = controller_nn(torch.cat((z, ey), dim=1))
        output = criterion(u_pred, u_target)
        # Compute a L1 norm loss to encourage tighter IBP bounds.
        lya_params = torch.cat([p.view(-1) for p in controller_nn.parameters()])
        l1_loss = l1_reg * torch.norm(lya_params, 1) / total_elements
        loss = output + l1_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller_nn.parameters(), 1)
        print(
            f"iter {i}, mse {output.item()}, l1 {l1_loss.item()}, loss {loss.item()}, lr {optimizer.param_groups[0]['lr']}"
        )
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            print("Best controller model save to ", file_name)
            torch.save(controller_nn.state_dict(), file_name)


def approximate_ekf(
    observer,
    ekf_observer,
    controller,
    nx,
    nA,
    x_max,
    e_max,
    A_max,
    h,
    file_name,
    batch_size=100,
    lr=1e-2,
    max_iter=500,
    l1_reg=1e-3,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(observer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)
    total_elements = sum(p.numel() for p in observer.parameters())
    best_loss = np.inf

    for i in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        x = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * x_max
        e = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * e_max
        P = ekf_observer.P0
        z = x - e
        y = h(x)
        u = controller(x)
        z_next = ekf_observer.forward_constant_K(z, u, y)
        z_pred = observer(z, u, y)
        output = criterion(z_next, z_pred)
        lya_params = torch.cat([p.view(-1) for p in observer.parameters()])
        l1_loss = l1_reg * torch.norm(lya_params, 1) / total_elements
        loss = output
        loss.backward()
        print(
            f"iter {i}, mse {output.item()}, l1 {l1_loss.item()}, loss {loss.item()}, lr {optimizer.param_groups[0]['lr']}"
        )
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            print("Best observer model save to ", file_name)
            torch.save(observer.state_dict(), file_name)


def fit_observer(
    observer,
    h,
    u,
    dynamics,
    nx,
    x_max,
    e_max,
    file_name,
    gamma=0,
    roll_out_steps=150,
    batch_size=100,
    lr=1e-3,
    max_iter=500,
    l1_reg=1e-3,
):
    print("Fitting observer function")
    optimizer = torch.optim.Adam(observer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)
    total_elements = sum(p.numel() for p in observer.parameters())
    best_loss = np.inf

    time_step = torch.arange(roll_out_steps).to(device)
    weight = torch.ones((batch_size, 1)).to(device) * gamma
    W = 1 - torch.pow(weight, time_step + 1).unsqueeze(2)
    W = W.repeat(1, 1, nx)

    for i in range(max_iter):
        optimizer.zero_grad()
        x = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * x_max
        e = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * e_max
        z = x - e
        x_data = []
        z_data = []
        # Roll out trajectory from suboptimal state feedback controller
        for _ in range(roll_out_steps):
            u_x = u(x)
            y = h(x)
            x = dynamics(x, u_x)
            z = observer(z, u_x, y)
            x_data.append(x)
            z_data.append(z)
        x_data = torch.stack(x_data)
        z_data = torch.stack(z_data)
        # Compute a L1 norm loss to encourage tighter IBP bounds.
        lya_params = torch.cat([p.view(-1) for p in observer.parameters()])
        l1_loss = l1_reg * torch.norm(lya_params, 1) / total_elements
        mse_loss = torch.mean(
            W * (x_data.transpose(0, 1) - z_data.transpose(0, 1)) ** 2
        )
        loss = mse_loss + l1_loss
        torch.nn.utils.clip_grad_norm_(observer.parameters(), 1)
        print(
            f"iter {i}, mse {mse_loss},l1 {l1_loss.item()}, loss {loss}, lr {optimizer.param_groups[0]['lr']}"
        )
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            print("Best observer model save to ", file_name)
            torch.save(observer.state_dict(), file_name)


def approximate_lyapunov_from_rollouts(
    lyapunov_nn,
    controller,
    dynamics,
    observer,
    nx,
    nA,
    nu,
    x_max,
    e_max,
    A_max,
    file_name,
    batch_size=100,
    roll_out_steps=150,
    lr=5e-3,
    max_iter=500,
    l1_reg=0.1,
):
    print("Approximating lyapunov function from rollouts...")

    Q = torch.eye(2 * nx, dtype=dtype, device=device) / 1e3
    R = torch.eye(nu, dtype=dtype, device=device) / 1e3
    u_eq = dynamics.u_equilibrium.to(device)

    def l(xe, u):
        x_cost = torch.einsum("bi, ij, bj->b", xe, Q, xe)
        u_cost = torch.einsum("bi, ij, bj->b", u - u_eq, R, u - u_eq)
        cost = x_cost + u_cost
        return cost

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lyapunov_nn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)
    total_elements = sum(p.numel() for p in lyapunov_nn.parameters())
    best_loss = np.inf
    # tril_indices = torch.tril_indices(row=nx, col=nx, offset=0)

    for i in range(max_iter):
        optimizer.zero_grad()
        # Sample x and compute target.
        x = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * x_max
        e = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * e_max
        # A_vec = (torch.rand((batch_size, nA), device=device) - 0.5) *2 * A_max
        # A = torch.zeros(batch_size, nx, nx, device=device)
        # A[:, tril_indices[0], tril_indices[1]] = A_vec
        # P = A @ A.transpose(1, 2)
        z = x - e
        V_data = []
        stage_cost = []
        for _ in range(roll_out_steps):
            u_z = controller(z)
            y = observer.h(x)
            x = dynamics(x, u_z)
            z = observer.forward(z, u_z, y)
            xe = torch.hstack((x, x - z))
            V_data.append(lyapunov_nn(xe))
            stage_cost.append(l(xe, u_z))
        V_data = torch.stack(V_data).squeeze().transpose(0, 1)
        stage_cost_flip = torch.stack(stage_cost).transpose(0, 1).fliplr()
        J_pi = stage_cost_flip.cumsum(dim=1).fliplr()
        output = criterion(J_pi, V_data)
        # Compute a L1 norm loss to encourage tighter IBP bounds.
        lya_params = torch.cat([p.view(-1) for p in lyapunov_nn.parameters()])
        l1_loss = l1_reg * torch.norm(lya_params, 1) / total_elements
        loss = output + l1_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lyapunov_nn.parameters(), 1)
        print(
            f"iter {i}, mse {output.item()}, l1 {l1_loss.item()}, loss {loss.item()}, lr {optimizer.param_groups[0]['lr']}"
        )
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            print("Best lyapunov model save to ", file_name)
            torch.save(lyapunov_nn.state_dict(), file_name)


def approximate_lyapunov_nn(
    lyapunov_target,
    lyapunov_nn,
    nx,
    nA,
    x_max,
    e_max,
    A_max,
    file_name,
    batch_size=100,
    lr=0.05,
    max_iter=500,
    l1_reg=0.05,
):
    print("Approximating Lyapunov neural network with smaller NN")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lyapunov_nn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)
    total_elements = sum(p.numel() for p in lyapunov_nn.parameters())
    best_loss = np.inf

    for i in range(max_iter):
        optimizer.zero_grad()
        x = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * x_max
        e = (torch.rand((batch_size, nx), device=device) - 0.5) * 2 * e_max
        # A = (torch.rand((batch_size, nA), device=device) - 0.5) *2 * A_max
        xe = torch.cat((x, e), dim=1)
        target = lyapunov_target(xe)
        mse = criterion(target, lyapunov_nn(xe))
        # Compute a L1 norm loss to encourage tighter IBP bounds.
        lya_params = torch.cat([p.view(-1) for p in lyapunov_nn.parameters()])
        l1_loss = l1_reg * torch.norm(lya_params, 1) / total_elements
        loss = mse + l1_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lyapunov_nn.parameters(), 1)
        print(
            f"iter {i}, mse {mse.item()}, l1 {l1_loss.item()}, loss {loss.item()}, lr {optimizer.param_groups[0]['lr']}"
        )
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            print("Best lyapunov model save to ", file_name)
            torch.save(lyapunov_nn.state_dict(), file_name)


def simulate(lyaloss: lyapunov.LyapunovDerivativeDOFLoss, steps: int, x0, z0):
    # Assumes explicit euler integration.
    x_traj = [None] * steps
    z_traj = [None] * steps
    V_traj = [None] * steps
    x_traj[0] = x0
    z_traj[0] = z0
    with torch.no_grad():
        V_traj[0] = lyaloss.lyapunov(
            torch.cat((x_traj[0], x_traj[0] - z_traj[0]), dim=1)
        )
        for i in range(1, steps):
            y = lyaloss.observer.h(x_traj[i - 1])
            z = z_traj[i - 1]
            u = lyaloss.controller.forward(
                torch.cat((z, y - lyaloss.observer.h(z)), dim=1)
            )
            x_traj[i] = lyaloss.dynamics.forward(x_traj[i - 1], u)
            z_traj[i] = lyaloss.observer.forward(z, u, y)
            V_traj[i] = lyaloss.lyapunov(
                torch.cat((x_traj[i], x_traj[i] - z_traj[i]), dim=1)
            )

    return (
        torch.stack(x_traj).cpu().detach().numpy(),
        torch.stack(z_traj).cpu().detach().numpy(),
        torch.stack(V_traj).cpu().detach().numpy().squeeze(),
    )
