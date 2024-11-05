#!/usr/bin/env python

"""
Generates documentation for the current package. Run with `make_docs` after
installing this package in a virtual environment.
Will generate documentation for every top-level folder except "resources".
Must live in the package_name module so that the alias can be installed.
"""

import os
import argparse
import subprocess

script = """
source "%s/bin/activate" &&
pip install pdoc -q &&
pdoc -t docs/theme -o docs csi_analysis examples tests &&
echo "Successfully generated documentation at $(pwd)/docs."
"""


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate documentation for the current package."
    )
    parser.add_argument(
        "--venv",
        help="The path to the virtual environment where pdoc is installed. "
        "Relative paths are relative to the repository root.",
        type=str,
        default=".venv",
    )
    return parser.parse_args()


def main():
    args = argument_parser()
    repository_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Check if args.venv is absolute or relative
    if not os.path.isabs(args.venv):
        args.venv = os.path.join(repository_path, args.venv)
    if not os.path.isdir(args.venv):
        raise FileNotFoundError(f"Virtual environment not found at {args.venv}.")
    subprocess.run(
        script % args.venv, shell=True, executable="/bin/bash", cwd=repository_path
    )


if __name__ == "__main__":
    main()
