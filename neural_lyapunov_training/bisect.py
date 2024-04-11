"""Bisection on rho."""
import argparse
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from complete_verifier import ABCROWN


def check_rho(rho):
    print(f"Generating specs with rho={rho}")
    output_gen_spec = os.path.join(
        args.output_folder, f"rho_{rho:.5f}_spec.txt")
    command = (
        "python -m neural_lyapunov_training.generate_vnnlib "
        f"--lower_limit {' '.join(map(str, args.lower_limit))} "
        f"--upper_limit {' '.join(map(str, args.upper_limit))} "
        f"--hole_size {args.hole_size} "
        f"--value_levelset {rho} "
    )
    if args.check_x_next_only:
        command += "--check_x_adv_only "
    command += f"{args.spec_prefix} >{output_gen_spec} 2>&1"
    os.system(command)
    print("Start verification")
    output_path = os.path.join(args.output_folder, f"rho_{rho:.5f}.txt")
    print("Output path:", output_path)
    with open(output_path, "w") as file:
        with redirect_stdout(file), redirect_stderr(file):
            verifier = ABCROWN(
                args=additional_args,
                csv_name=f'{args.spec_prefix}.csv',
                config=args.config,
                override_timeout=args.timeout,
                pgd_order="before",
                pgd_restarts=10000
            )
            ret = verifier.main()
    print("Result:", ret)
    result = "safe"
    for k, v in ret.items():
        if "unsafe" in k:
            result = "unsafe"
    if result == "safe" and "unknown" in ret.keys():
        result = "unknown"
    print(result)
    print()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec_prefix",
        type=str,
        default="specs/bisect",
        help="Filename prefix for the specs.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Folder for the output.",
    )
    parser.add_argument(
        "-l",
        "--lower_limit",
        type=float,
        nargs="+",
        help="Lower limit of state dimension. A list of state_dim numbers.",
    )
    parser.add_argument(
        "-u",
        "--upper_limit",
        type=float,
        nargs="+",
        help="Upper limit of state dimension. A list of state_dim numbers.",
    )
    parser.add_argument(
        "-o",
        "--hole_size",
        type=float,
        default=0.001,
        help="Relative size of the hole in the middle to skip verification (0.0 - 1.0).",
    )
    parser.add_argument(
        "--init_rho",
        type=float,
        default=None,
        required=True,
        help="Initial rho value."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Configuration file for verification."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=200,
        help="Timeout for running verification and attack."
    )
    parser.add_argument(
        "--rho_eps",
        type=float,
        default=0.001,
        help="Precision of the rho bisection."
    )
    parser.add_argument(
        "--rho_multiplier",
        type=float,
        default=1.2,
        help="Multiplier for enlarging rho."
    )
    parser.add_argument(
        "--check_x_next_only",
        action="store_true",
        help="Only check the x_next condition but not the dV condition."
    )
    args, additional_args = parser.parse_known_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    rho = args.init_rho
    ret_init = check_rho(rho)
    if ret_init == "safe":
        while ret_init == "safe":
            rho *= args.rho_multiplier
            ret_init = check_rho(rho)
        rho_l = rho / args.rho_multiplier
        rho_u = rho
    else:
        while ret_init != "safe":
            rho /= args.rho_multiplier
            ret_init = check_rho(rho)
        rho_l = rho
        rho_u = rho * args.rho_multiplier

    while rho_u - rho_l > args.rho_eps:
        rho_m = (rho_l + rho_u) / 2
        print(f"rho_l={rho_l}, rho_u={rho_u}, rho_m={rho_m}")
        if check_rho(rho_m) == "safe":
            rho_l = rho_m
        else:
            rho_u = rho_m

    print(f"rho_l={rho_l}")
    print(f"rho_u={rho_u}")
