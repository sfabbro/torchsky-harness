"""Prototype side of compare_product.py, step 1: fit the truth problem (items, 8 exposures, noise seed 0) with the prototype and save the state into
port_check/truth_states (the prototype tree is not written to). Same two commands as tests/truth/uncertainty_calibration.py 'kernel' and 'fit'.

    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python proto_truth_fit.py
"""
import argparse
import os
import sys

PROTO = "/home/claude/work/fieldop_min"
sys.path.insert(0, PROTO)
sys.path.insert(0, os.path.join(PROTO, "tests", "truth"))
import uncertainty_calibration as uc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truth_states")
uc.OUT = OUT
uc.KERNEL = os.path.join(OUT, "kernel_truth.pt")
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(uc.KERNEL):
    uc.cmd_kernel(argparse.Namespace())
uc.cmd_fit(argparse.Namespace(form="items", noise_seed=0, fit_seed=0, half=-1, n_train=8, steps=0))
