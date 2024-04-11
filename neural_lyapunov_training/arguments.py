#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
Arguments parser and config file loader.

When adding new commandline parameters, please make sure to provide a clear and descriptive help message and put it in under a related hierarchy.
"""

import re
import os
from secrets import choice
import sys
import yaml
import time
import argparse
from collections import defaultdict


class ConfigHandler:
    def __init__(self):
        self.config_file_hierarchies = {
            # Given a hierarchy for each commandline option. This hierarchy is used in yaml config.
            # For example: "batch_size": ["solver", "propagation", "batch_size"] will be an element in this dictionary.
            # The entries will be created in add_argument() method.
        }
        # Stores all arguments according to their hierarchy.
        self.all_args = {}
        # Parses all arguments with their defaults.
        self.defaults_parser = argparse.ArgumentParser()
        # Parses the specified arguments only. Not specified arguments will be ignored.
        self.no_defaults_parser = argparse.ArgumentParser(
            argument_default=argparse.SUPPRESS
        )
        # Help message for each configuration entry.
        self.help_messages = defaultdict(str)
        # Add all common arguments.
        self.add_common_options()

    def add_common_options(self):
        """
        Add all parameters that are shared by different front-ends.
        """

        # We must set how each parameter will be presented in the config file, via the "hierarchy" parameter.
        # Global Configurations, not specific for a particular algorithm.

        # The "--config" option does not exist in our parameter dictionary.
        self.add_argument(
            "--config",
            type=str,
            help="Path to YAML format config file.",
            hierarchy=None,
        )

        h = ["general"]
        self.add_argument(
            "--seed",
            type=int,
            default=1234,
            help="Random seed.",
            hierarchy=h + ["seed"],
        )
        self.add_argument(
            "--dump_path",
            type=str,
            default=None,
            help="Dump config to this file.",
            hierarchy=h + ["dump_path"],
        )

        h = ["model"]
        self.add_argument(
            "--load_lyaloss",
            type=str,
            default=None,
            help="Path to load lyaloss state dict.",
            hierarchy=h + ["load_lyaloss"],
        )
        self.add_argument(
            "--save_lyaloss",
            action="store_true",
            help="Save lyaloss state dict file name.",
            hierarchy=h + ["save_lyaloss"],
        )
        self.add_argument(
            "--limit_scale",
            type=float,
            default=0.1,
            help="Scaling of state box.",
            hierarchy=h + ["limit_scale"],
        )
        self.add_argument(
            "--kappa",
            type=float,
            default=0.01,
            help="Lyapunov exponential decay rate.",
            hierarchy=h + ["kappa"],
        )
        self.add_argument(
            "--V_decrease_within_roa",
            action="store_true",
            help="Only requires V to decrease within the certified ROA (the sub-level set).",
            hierarchy=h + ["V_decrease_within_roa"],
        )
        self.add_argument(
            "--V_psd_form",
            type=str,
            default="L1",
            help="V_psd part form, L1 or quadratic.",
            hierarchy=h + ["V_psd_form"],
        )

        h = ["train"]
        self.add_argument(
            "--train_lyaloss",
            action="store_true",
            help="Train Lyapunov, controller and (maybe) observer.",
            hierarchy=h + ["train_lyaloss"],
        )
        self.add_argument(
            "--enable_wandb",
            action="store_true",
            help="Enable wandb.",
            hierarchy=h + ["enable_wandb"],
        )
        self.add_argument(
            "--lr_scheduler",
            action="store_true",
            help="Enable learning rate scheduler.",
            hierarchy=h + ["lr_scheduler"],
        )
        self.add_argument(
            "--max_iter",
            type=int,
            default=100,
            help="Number of outermost iterations in training.",
            hierarchy=h + ["max_iter"],
        )
        self.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="Learning rate.",
            hierarchy=h + ["learning_rate"],
        )
        self.add_argument(
            "--pgd_steps",
            type=int,
            default=50,
            help="Number of steps in pgd attack.",
            hierarchy=h + ["pgd_steps"],
        )
        self.add_argument(
            "--buffer_size",
            type=int,
            default=1000,
            help="Size of the state buffer.",
            hierarchy=h + ["buffer_size"],
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=100,
            help="Size of each batch.",
            hierarchy=h + ["batch_size"],
        )
        self.add_argument(
            "--epochs",
            type=int,
            default=40,
            help="Num epochs.",
            hierarchy=h + ["epochs"],
        )
        self.add_argument(
            "--samples_per_iter",
            type=int,
            default=100,
            help="Number of samples per PGD attack.",
            hierarchy=h + ["samples_per_iter"],
        )
        self.add_argument(
            "--Vmin_x_pgd_buffer_size",
            type=int,
            default=500000,
            help="Size of Vmin_x_pgd buffer.",
            hierarchy=h + ["Vmin_x_pgd_buffer_size"],
        )
        self.add_argument(
            "--derivative_x_buffer_path",
            type=str,
            default=None,
            help="Path to load a dataset of x as potential adversarial examples for the derivative condition.",
            hierarchy=h + ["derivative_x_buffer_path"],
        )
        self.add_argument(
            "--Vmin_x_pgd_path",
            type=str,
            default=None,
            help="Path to load a dataset of x that potentially minimizes V(x) on the boundary.",
            hierarchy=h + ["Vmin_x_pgd_path"],
        )

        h = ["loss"]
        self.add_argument(
            "--ibp_ratio_derivative",
            type=float,
            default=0.0,
            help="Ratio of IBP loss.",
            hierarchy=h + ["ibp_ratio_derivative"],
        )
        self.add_argument(
            "--sample_ratio_derivative",
            type=float,
            default=1.0,
            help="Ratio of sample loss.",
            hierarchy=h + ["sample_ratio_derivative"],
        )
        self.add_argument(
            "--ibp_ratio_positivity",
            type=float,
            default=0.0,
            help="Ratio of IBP loss.",
            hierarchy=h + ["ibp_ratio_positivity"],
        )
        self.add_argument(
            "--sample_ratio_positivity",
            type=float,
            default=0.0,
            help="Ratio of sample loss.",
            hierarchy=h + ["sample_ratio_positivity"],
        )
        self.add_argument(
            "--Vmin_x_boundary_weight",
            type=float,
            default=0.0,
            help="Weight for the regulizer min V(x) on boundary.",
            hierarchy=h + ["Vmin_x_boundary_weight"],
        )
        self.add_argument(
            "--Vmax_x_boundary_weight",
            type=float,
            default=0.0,
            help="Weight for the regulizer max V(x) on boundary.",
            hierarchy=h + ["Vmax_x_boundary_weight"],
        )
        self.add_argument(
            "--l1_reg",
            type=float,
            default=1e-3,
            help="Weight for L1norm(θ) in the loss.",
            hierarchy=h + ["l1_reg"],
        )
        self.add_argument(
            "--candidate_roa_states_weight",
            type=float,
            default=5e-4,
            help="Weight for candidate states in ROA.",
            hierarchy=h + ["candidate_roa_states_weight"],
        )

    def add_argument(self, *args, **kwargs):
        """Add a single parameter to the parser. We will check the 'hierarchy' specified and then pass the remaining arguments to argparse."""
        if "hierarchy" not in kwargs:
            raise ValueError(
                "please specify the 'hierarchy' parameter when using this function."
            )
        hierarchy = kwargs.pop("hierarchy")
        help = kwargs.get("help", "")
        private_option = kwargs.pop("private", False)
        # Make sure valid help is given
        if not private_option:
            if len(help.strip()) < 10:
                raise ValueError(
                    f'Help message must not be empty, and must be detailed enough. "{help}" is not good enough.'
                )
            elif (not help[0].isupper()) or help[-1] != ".":
                raise ValueError(
                    f'Help message must start with an upper case letter and end with a dot (.); your message "{help}" is invalid.'
                )
        self.defaults_parser.add_argument(*args, **kwargs)
        # Build another parser without any defaults.
        if "default" in kwargs:
            kwargs.pop("default")
        self.no_defaults_parser.add_argument(*args, **kwargs)
        # Determine the variable that will be used to save the argument by argparse.
        if "dest" in kwargs:
            dest = kwargs["dest"]
        else:
            dest = re.sub("^-*", "", args[-1]).replace("-", "_")
        # Also register this parameter to the hierarchy dictionary.
        self.config_file_hierarchies[dest] = hierarchy
        if hierarchy is not None and not private_option:
            self.help_messages[",".join(hierarchy)] = help

    def set_dict_by_hierarchy(self, args_dict, h, value, nonexist_ok=True):
        """Insert an argument into the dictionary of all parameters. The level in this dictionary is determined by list 'h'."""
        # Create all the levels if they do not exist.
        current_level = self.all_args
        assert len(h) != 0
        for config_name in h:
            if config_name not in current_level:
                if nonexist_ok:
                    current_level[config_name] = {}
                else:
                    raise ValueError(f"Config key {h} not found!")
            last_level = current_level
            current_level = current_level[config_name]
        # Add config value to leaf node.
        last_level[config_name] = value

    def construct_config_dict(self, args_dict, nonexist_ok=True):
        """Based on all arguments from argparse, construct the dictionary of all parameters in self.all_args."""
        for arg_name, arg_val in args_dict.items():
            h = self.config_file_hierarchies[arg_name]  # Get levels for this argument.
            if h is not None:
                assert len(h) != 0
                self.set_dict_by_hierarchy(
                    self.all_args, h, arg_val, nonexist_ok=nonexist_ok
                )

    def update_config_dict(self, old_args_dict, new_args_dict, levels=None):
        """Recursively update the dictionary of all parameters based on the dict read from config file."""
        if levels is None:
            levels = []
        if isinstance(new_args_dict, dict):
            # Go to the next dict level.
            for k in new_args_dict:
                self.update_config_dict(
                    old_args_dict, new_args_dict[k], levels=levels + [k]
                )
        else:
            # Reached the leaf level. Set the corresponding key.
            self.set_dict_by_hierarchy(
                old_args_dict, levels, new_args_dict, nonexist_ok=False
            )

    def dump_config(self, args_dict, level=[], out_to_doc: str = None, show_help=False):
        """Generate a config file based on args_dict with help information.

        Args:
          out_to_doc The path of the file where we dump the config.
        """
        ret_string = ""
        for key, val in args_dict.items():
            if isinstance(val, dict):
                ret = self.dump_config(val, level + [key], out_to_doc, show_help)
                if len(ret) > 0:
                    # Next level is not empty, print it.
                    ret_string += " " * (len(level) * 2) + f"{key}:\n" + ret
            else:
                if show_help:
                    h = self.help_messages[",".join(level + [key])]
                    if (
                        "debug" in key
                        or "not use" in h
                        or "not be use" in h
                        or "debug" in h
                        or len(h) == 0
                    ):
                        # Skip some debugging options.
                        continue
                    h = f"  # {h}"
                else:
                    h = ""
                yaml_line = (
                    yaml.safe_dump({key: val}, default_flow_style=None)
                    .strip()
                    .replace("{", "")
                    .replace("}", "")
                )
                ret_string += " " * (len(level) * 2) + f"{yaml_line}{h}\n"
        if len(level) > 0:
            return ret_string
        else:
            # Top level, output to file.
            if out_to_doc:
                with open(out_to_doc, "w") as f:
                    f.write(ret_string)
            return ret_string

    def parse_config(self):
        """
        Main function to parse parameter configurations. The commandline arguments have the highest priority;
        then the parameters specified in yaml config file. If a parameter does not exist in either commandline
        or the yaml config file, we use the defaults defined in add_common_options() defined above.
        """
        # Parse an empty commandline to get all default arguments.
        default_args = vars(self.defaults_parser.parse_args([]))
        # Create the dictionary of all parameters, all set to their default values.
        self.construct_config_dict(default_args)
        # Update documents.
        # self.dump_config(self.all_args, out_to_doc=True, show_help=True)
        # These are arguments specified in command line.
        specified_args = vars(self.no_defaults_parser.parse_args())
        # Read the yaml config files.
        if "config" in specified_args:
            with open(specified_args["config"], "r") as config_file:
                loaded_args = yaml.safe_load(config_file)
                # Update the defaults with the parameters in the config file.
                self.update_config_dict(self.all_args, loaded_args)
        # Finally, override the parameters based on commandline arguments.
        self.construct_config_dict(specified_args, nonexist_ok=False)
        # For compatibility, we still return all the arguments from argparser.
        parsed_args = self.defaults_parser.parse_args()
        # Print all configuration.
        print("Configurations:\n")
        print(self.dump_config(self.all_args))
        return parsed_args

    def keys(self):
        return self.all_args.keys()

    def items(self):
        return self.all_args.items()

    def __getitem__(self, key):
        """Read an item from the dictionary of parameters."""
        return self.all_args[key]

    def __setitem__(self, key, value):
        """Set an item from the dictionary of parameters."""
        self.all_args[key] = value


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("You must register a global parameter in arguments.py.")

    def __setitem__(self, key, value):
        if key not in self:
            raise RuntimeError("You must register a global parameter in arguments.py.")
        else:
            super().__setitem__(key, value)

    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


# Global configuration variable
Config = ConfigHandler()
# Global variables
Globals = ReadOnlyDict(
    {
        "starting_timestamp": int(time.time()),
        "example_idx": -1,
        "lp_perturbation_eps": None,
    }
)
