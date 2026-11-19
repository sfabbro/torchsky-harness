"""Markdown tables of the reproducer runs (results/*.json) for SOLVER_DIAGNOSIS_2026-11-19.md.

    python tests/solver_repro/summarise.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def load(n):
    p = os.path.join(R, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def first_below(h, t):
    return next((i + 1 for i, v in enumerate(h) if v < t), None)


def row(name, d, label):
    h = d["polish"]["history"]
    at = lambda i: f"{h[i - 1]:.1e}" if i <= len(h) else "-"  # noqa: E731
    rz = d["polish"]["ritz"]
    it3 = first_below(h, 1e-3)
    return f"| {label} | {name} | {at(10)} | {at(30)} | {at(60)} | {at(100)} | {it3 if it3 else '>100'} | {min(rz):.3f} to {max(rz):.3g} |"


ISO = [("six_plain", "six filters, K = 12, two channels; no gap, no bright sources, no floor, no Huber"),
       ("six_gap", "+ chip gap (dithers fill it)"),
       ("six_gapc", "+ chip gap, F090W's gap on the box centre"),
       ("six_bright", "six filters, no gap; bright sources (peak S/N 3000) and floor 9 / 4 per cent"),
       ("six_bright_huber", "+ Huber weights"),
       ("one_all", "ONE filter (K = 3), all other features"),
       ("onechan_all", "all features, one channel (all six filters at 0.031 arcsec)"),
       ("items_all", "all features, item sky"),
       ("all", "all features (gap, bright, floor, Huber, two channels)")]


def main():
    print("| configuration | preconditioner | 10 | 30 | 60 | 100 | iterations to 1e-3 | Lanczos range |")
    print("|---|---|---|---|---|---|---|---|")
    for key, label in ISO:
        for m, nm in (("old", "fourier (old)"), ("filter", "filter (new)")):
            d = load(f"iso_{key}_{m}")
            if d:
                print(row(nm, d, label))
    print()
    for n in ("base_one_old", "all_old", "all_filterq", "gapc_old", "gapc_filterq", "all_filter", "all_filter6", "gapc_filter6"):
        d = load(n)
        if d:
            print(row(n, d, "draw" if False else "polish"), "draw at 60/100:",
                  [f"{d['draw']['history'][i - 1]:.1e}" for i in (60, 100) if "draw" in d and i <= len(d["draw"]["history"])])


if __name__ == "__main__":
    main()
