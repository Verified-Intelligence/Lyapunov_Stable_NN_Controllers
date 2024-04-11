# Example of generating a grid of points and verify the grid in a batch using autoLiRPA library (incomplete verification).

import sys
from math import pi
import random
import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import (
    BoundMul,
    BoundTanh,
    BoundSigmoid,
    BoundSin,
    BoundCos,
    BoundTan,
    BoundActivation,
)
import matplotlib.pyplot as plt
from tqdm import trange
from neural_lyapunov_training.models import (
    create_acrobot_model,
    create_pendulum_model,
    create_quadrotor2d_model,
    create_quadrotor3d_model,
)
from neural_lyapunov_training.train_utils import generate_grids

seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# loss = create_quadrotor3d_model()
# loss = create_acrobot_model()
# loss = create_pendulum_model()
loss = create_quadrotor2d_model()
grid_size = 1


def test_input(x):
    print(f"input: {x}")
    print(f"controller: {loss.controller(x)}")
    print(f"dynamics: {loss.dynamics(x=x, u=loss.controller(x))}")
    print(f"lyapunov: {loss.lyapunov(x)}")
    print(f"loss: {loss(x)}")


x_dim = loss.dynamics.x_equilibrium.size(0)
print(loss(torch.randn(100, x_dim)).squeeze())
x = torch.ones(1, x_dim)
test_input(x)
x = torch.zeros(1, x_dim)
test_input(x)

lower_limit = -1.0 * torch.ones(x_dim)
upper_limit = 1.0 * torch.ones(x_dim)
grid_size = grid_size * torch.ones(x_dim, dtype=torch.int32)
lower, upper = generate_grids(lower_limit, upper_limit, grid_size)
x = (lower + upper) / 2.0
print(f"shape of x: {x.size()}")

bounded_lyapunov = BoundedModule(
    loss, x, bound_opts={"conv_mode": "matrix", "sparse_intermediate_bounds": False}
)
eps = None
norm = np.inf
ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=lower, x_U=upper)
bounded_state = BoundedTensor(x, ptb)

with torch.no_grad():
    pred = bounded_lyapunov(bounded_state)
    lb, ub = bounded_lyapunov.compute_bounds(x=(bounded_state,), method="CROWN")

for m in bounded_lyapunov.modules():
    if isinstance(m, BoundActivation) and hasattr(m, "lower"):
        l = m.lower
        u = m.upper
        d = (l - u).abs().sum()
        print(f"{m} {d}")

# import pdb; pdb.set_trace()
print(pred.view(-1).numpy())
print(lb.view(-1).numpy())
print(ub.view(-1).numpy())
