import sys
import os
import argparse
import json
import torch
import numpy as np

sys.path = ["complete_verifier"] + sys.path
from arguments import ConfigHandler
from abcrown import ABCROWN

parser = argparse.ArgumentParser()
parser.add_argument(
    "config", type=str, default="Path to the config file for the verification."
)
parser.add_argument("model", type=str, default="Path to the Lyapunov model checkpoint.")
parser.add_argument("--timeout", type=int, default=10)
parser.add_argument("--decision_thresh", type=float, default=1000)
parser.add_argument("--batch_size", type=int, default=262144)
parser.add_argument('--lyapunov', type=str, default=None,
                    help='Directly provide a definition to the Lyapunov function.')
args = parser.parse_args()

config = ConfigHandler()
config.parse_config([f"--config={args.config}"], verbose=False)
input_dim = config["model"]["input_shape"][-1]
print("Input dimension:", input_dim)

if args.lyapunov:
    model = args.lyapunov
else:
    model_params = config["model"]["name"]
    assert "lyapunov_parameters" in model_params

    model_params_ = model_params[model_params.find("lyapunov_parameters") :]
    lyapunov_params = model_params_[model_params_.find("{") : model_params_.find("}") + 1]
    print("Lyapunov params:", lyapunov_params)
    lyapunov_params = eval(lyapunov_params)
    lyapunov_params_ = ""
    assert isinstance(lyapunov_params, dict)
    for k, v in lyapunov_params.items():
        if v is None or isinstance(v, (int, bool, list, float)):
            lyapunov_params_ += f"{k}={v}, "
        elif isinstance(v, str):
            lyapunov_params_ += f'{k}="{v}", '
        elif v == torch.nn.LeakyReLU:
            lyapunov_params_ += f"{k}=torch.nn.LeakyReLU, "
        else:
            raise NotImplementedError(f"Unsupported Lyapunov parameter: {k}={v}")

    if 'lyapunov_func="lyapunov.NeuralNetworkQuadraticLyapunov"' in model_params:
        print("Using quadratic Lyapunov")
        model = (
            'Customized("neural_lyapunov_training/lyapunov.py", "NeuralNetworkQuadraticLyapunov",'
            f" x_dim={input_dim},"
            f" goal_state=torch.zeros({input_dim}),"
            f" {lyapunov_params_})"
        )
    else:
        print("Using NN Lyapunov")
        model = (
            'Customized("neural_lyapunov_training/lyapunov.py", "NeuralNetworkLyapunov",'
            f" x_dim={input_dim},"
            f" goal_state=torch.zeros({input_dim}),"
            f" {lyapunov_params_})"
        )

abcrown_args = [
    f"--config={args.config}",
    f"--override_timeout={args.timeout}",
    f"--decision_thresh={args.decision_thresh}",
    f"--load_model={args.model}",
    f"--model={model}",
    "--csv_name",
    "",
    "--num_outputs=1",
    "--sort_domain_interval=1",
    "--spec_type=box",
    "--robustness_type=all-positive",
    "--enable_input_split",
    "--branching_method=sb",
    "--sb_coeff_thresh=0.01",
]
if args.batch_size:
    abcrown_args.append(f"--batch_size={args.batch_size}")
verifier = ABCROWN(abcrown_args)

print("Running abcrown to get the level set")
verifier.main()
print("\n\n\n")

levelset = float("inf")
for item in verifier.logger.bab_ret:
    levelset = min(levelset, args.decision_thresh + item[1])
print("Level set:", levelset)
