# Generate VNNLIB specification used for verifying Lyapunov condition under level set constraint.
# Usage:
#     mkdir specs
#     python3 generate_vnnlib.py --lower_limit -1.57 -3.14 --upper_limit 1.57 3.14 --hole_size 0.001 --value_levelset 0.0455696 specs/pendulum_lyapunov_in_levelset

import os
import sys
import time
import socket
import argparse
from neural_lyapunov_training.models import box_data


def generate_preamble(out, state_dim, args):
    if abs(args.value_levelset) > 1e-10:
        out.write(
            "; VNNLIB Property for the verification of Lyapunov condition within a level set.\n\n"
        )
    else:
        out.write("; VNNLIB Property for the verification of Lyapunov condition.\n\n")
    try:
        user = os.getlogin()
    except OSError:
        user = "UNKNOWN_USER"
    out.write(
        f"; Generated at {time.ctime()} on {socket.gethostname()} by {user}\n"
    )
    out.write(f'; Generation command: \n; {" ".join(sys.argv)}\n\n')
    out.write("; Input variables (states).\n")

    for i in range(state_dim):
        out.write(f"(declare-const X_{i} Real)\n")

    if abs(args.value_levelset) > 1e-10:
        out.write(
            "\n; Output variables (Lyapunov condition, and Lyapunov function value).\n"
        )
    else:
        out.write("\n; Output variables (Lyapunov condition).\n")
    out.write("(declare-const Y_0 Real)\n")
    if abs(args.value_levelset) >= 1e-10:
        out.write("(declare-const Y_1 Real)\n")
        if not args.no_check_x_next:
            for i in range(2, 2 + state_dim):
                out.write(f"(declare-const Y_{i} Real)\n")
    else:
        assert args.no_check_x_next
    out.write("\n")


def generate_limits(out, lower_limit, upper_limit):
    assert len(lower_limit) == len(upper_limit)
    out.write("; Input constraints.\n\n")
    for i, (l, u) in enumerate(zip(lower_limit, upper_limit)):
        out.write(f"; Input state {i}.\n")
        out.write(f"(assert (<= X_{i} {u}))\n")
        out.write(f"(assert (>= X_{i} {l}))\n\n")


def generate_specs(out, args, lower_limit, upper_limit):
    if abs(args.value_levelset) > 1e-10:
        out.write("; Verifying Lyapunov condition (output 0) holds (positive), and\n")
        out.write("; Lyapunov function (output 1) is less than the level set value.\n")
        if not args.no_check_x_next:
            out.write(f"(assert (or\n")
            if not args.check_x_next_only:
                out.write(f"  (and (<= Y_0 -{args.tolerance}))\n")
            for i, (l, u) in enumerate(zip(args.lower_limit, args.upper_limit)):
                # "Y_{i+2}"" is X_next[i]
                out.write(f"  (and (<= Y_{i+2} {l - args.tolerance}))\n")
                out.write(f"  (and (>= Y_{i+2} {u + args.tolerance}))\n")
            out.write("))\n")
        else:
            out.write(f"(assert (<= Y_0 -{args.tolerance}))\n")
        out.write(f"(assert (<= Y_1 {args.value_levelset}))\n")
    else:
        out.write("; Verifying Lyapunov condition (output 0) holds (positive).\n")
        out.write(f"(assert (<= Y_0 -{args.tolerance}))\n")
        assert args.no_check_x_next


def generate_csv(num_regions, args, relpath=False):
    fname = args.output_filename + ".csv"
    with open(fname, "w") as out:
        print(f"Generating {fname}")
        for i in range(num_regions):
            if relpath:
                # Relative path to VNNLIB files in CSV file, e.g., file_1.vnnlib
                out.write(f"{os.path.basename(args.output_filename)}_{i}.vnnlib\n")
            else:
                # Full path to VNNLIB files in CSV file, e.g., specs/file_1.vnnlib
                out.write(f"{args.output_filename}_{i}.vnnlib\n")
    print(f"Done. Now change your verification config file to verify {fname}.")


def main():
    parser = argparse.ArgumentParser(
        prog="VNNLIB Generator",
        description="Generate VNNLIB property file for verification of Lyapunov condition under level set constraint",
    )
    parser.add_argument(
        "output_filename",
        type=str,
        help="Output filename prefix. A single csv file and multiple VNNLIB files will be generated.",
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
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help="Scaling of lower limit and upper limit. Used for quickly try different sizes.",
    )
    parser.add_argument(
        "-o",
        "--hole_size",
        type=float,
        default=0,
        help="Relative size of the hole in the middle to skip verification (0.0 - 1.0).",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for verification. For single precision it is around 1e-6.",
    )
    parser.add_argument(
        "-v",
        "--value_levelset",
        type=float,
        default=0.0,
        help="Level set value. We verify Lyapunov condition only when Lyapunov function is smaller than this value. Ignored when set to 0.",
    )
    parser.add_argument(
        "-r",
        "--relative_vnnlib_path",
        action="store_true",
        help="When specified, the vnnlib file path in CSV file will be relative to the path of the CSV file.",
    )
    parser.add_argument(
        "--no_check_x_next",
        action="store_true",
        help="Do not check if x_next is within the bounding box."
    )
    parser.add_argument(
        "--check_x_next_only",
        action="store_true",
        help="Only check if x_next is within the bounding box."
    )

    args = parser.parse_args()
    assert args.hole_size >= 0 and args.hole_size <= 0.2
    assert len(args.lower_limit) == len(args.upper_limit)
    if args.check_x_next_only:
        args.no_check_x_next = False
    state_dim = len(args.lower_limit)

    # Obtain the number of regions (subproblems) to verify.
    _, _, data_max, data_min, _ = box_data(
        lower_limit=args.lower_limit,
        upper_limit=args.upper_limit,
        ndim=len(args.lower_limit),
        hole_size=args.hole_size,
        scale=args.scale,
    )
    num_regions = data_min.size(0)

    for i in range(num_regions):
        fname = f"{args.output_filename}_{i}.vnnlib"
        with open(fname, "w") as out:
            lower_limit, upper_limit = data_min[i].tolist(), data_max[i].tolist()
            print(
                f"Generating {fname} with lower_limit = {lower_limit}, upper_limit = {upper_limit}"
            )
            generate_preamble(out, state_dim, args)
            generate_limits(out, lower_limit, upper_limit)
            generate_specs(out, args, lower_limit, upper_limit)
    generate_csv(num_regions, args, relpath=args.relative_vnnlib_path)


if __name__ == "__main__":
    main()
