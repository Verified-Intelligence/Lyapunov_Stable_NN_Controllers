import torch
import sys
from matplotlib import cm
import matplotlib.pyplot as plt

device = 'cuda'

if sys.argv[1] == 'pendulum':
    lower_limit = torch.tensor([-12., -12.])
    upper_limit = torch.tensor([12., 12.])
    rho_ditl = 22.74517822265625
    rho_ours = 25.58

    from pendulum import LyapunovNet
    V = LyapunovNet().to(device)
    V.load_state_dict(torch.load('baselines/nlc_discrete/models/pendulum_lyapunov.pth'))
elif sys.argv[1] == 'path_tracking':
    lower_limit = torch.tensor([-3., -3.])
    upper_limit = torch.tensor([3., 3.])
    rho_ditl = 60.99
    rho_ours = 72.425

    from path_tracking import LyapunovNet
    V = LyapunovNet().to(device)
    V.load_state_dict(torch.load('baselines/nlc_discrete/models/path_tracking_lyapunov.pth'))
elif sys.argv[1] == 'cartpole':
    lower_limit = -torch.ones(4)
    upper_limit = torch.ones(4)
    rho_ditl = 4.40
    rho_ours = 4.915

    from cartpole import LyapunovNet
    V = LyapunovNet().to(device)
    V.load_state_dict(torch.load('baselines/nlc_discrete/models/cartpole_lyapunov.pth'))
else:
    raise NameError(sys.argv[1])

def plot_V_heatmap(V, lower_limit, upper_limit, rho_ours, rho_ditl):
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
    im = ax.pcolor(grid_x, grid_y, V_val, cmap=cm.coolwarm)
    ax.contour(grid_x, grid_y, V_val, [rho_ditl], colors="orange")
    ax.contour(grid_x, grid_y, V_val, [rho_ours], colors="black")
    ax.set_xlim(lower_limit[0], upper_limit[0])
    ax.set_ylim(lower_limit[1], upper_limit[1])
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.legend()
    cbar = fig.colorbar(im, ax=ax)
    return fig, ax, cbar

if sys.argv[1] == 'cartpole':
    raise NotImplementedError
fig, ax, cbar = plot_V_heatmap( V, lower_limit, upper_limit, rho_ours, rho_ditl)
fig.savefig(f'{sys.argv[1]}.pdf')
