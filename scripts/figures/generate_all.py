#!/usr/bin/env python
"""
Generate all figures for the H-NBDL paper.

Usage:
    python scripts/figures/generate_all.py --output figures/
    python scripts/figures/generate_all.py --figure 3  # single figure
"""

import argparse
from pathlib import Path

# Import from the main figure generation module
# (copy generate_figures.py content or import as module)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output", type=str, default="figures/")
    parser.add_argument("--figure", type=int, default=None, help="Generate a single figure (1-8)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures to {outdir}/")
    print("Run generate_figures.py from the project root for full figure generation.")
    print("See scripts/figures/generate_all.py for the complete pipeline.")


if __name__ == "__main__":
    main()
