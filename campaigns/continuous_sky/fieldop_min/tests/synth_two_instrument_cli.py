"""Two-instrument synthetic field through the whole command-line path (cli.main), with the exposures handed in by a stand-in loader.

Used twice (INSTRUMENT_INTERFACE.md, check 5): once on the package as it stood before the instrument interface (a copy outside the tree), once on
the current package; the held-out scores and the state hash must be identical. The stand-in loader returns the synthetic field of
tests/test_defaults_v2.py (instruments A and B); everything after loading (spectral domain, filter-curve grid, instrument constants, fit,
held-out stage) runs the package's own code.

    PYTHONNOUSERSITE=1 python tests/synth_two_instrument_cli.py --pkg PACKAGE_DIR --out RESULT.json
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

import torch

torch.set_num_threads(1)

OPTIONS = ["--sky-pitch", "0.08", "--render-pitch", "0.04", "--level-nh", "8", "--kernel-extent", "0.8", "--region-px", "12", "--n-lambda", "3",
           "--k-hidden", "16", "--stripes", "0", "--batch-patches", "3", "--tau-mode", "fixed", "--log-every", "2",
           "--held-steps", "6", "--held-reg-steps", "4", "--warm-kernel-steps", "5", "--reg-prepass-steps", "3", "--reg-prepass-iters", "1",
           "--budget-min", "100000", "--device", "cpu", "--steps", "6", "--gauge-ref-inst", "0", "--seed", "0"]


def run(pkg, workdir, extra=()):
    """Run cli.main of the package `pkg` on the synthetic field; return (held-out summary, per-exposure scores, state hash)."""
    sys.path.insert(0, os.path.abspath(pkg))
    cur = sys.modules.get("fieldop")
    if cur is None or os.path.dirname(os.path.abspath(cur.__file__)) != os.path.join(os.path.abspath(pkg), "fieldop"):
        for m in [k for k in sys.modules if k == "fieldop" or k.startswith("fieldop.")]:     # another package copy: import it afresh
            del sys.modules[m]
    import fieldop.cli as cli
    import fieldop.data as data
    from fieldop.synth import SynthInstrument, make_field
    insts = [SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]

    def fake_load(cfg, dev, log=print):
        exps, fset, meta, truth, _ = make_field(seed=1, n_per_filter=4, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3),
                                                patch_px=12, truth_half=1.0)
        for i, e in enumerate(exps):
            e.cid = i
        info = dict(box={}, boxes=[{}], tiles=[0], tile_c0={0: [0, 0]}, half_data=1.9, dets=list(range(meta["n_det"])),
                    chans=[f"c{i}" for i in range(meta["n_chan"])], insts=["A", "B"], filters=list(fset.names),
                    n_by_split={s: sum(1 for e in exps if e.split == s) for s in ("train", "held")}, shifts_applied=0,
                    n_exposures=len(exps), n_cuts=len(exps), cuts_per_tile={0: {}}, host_gb=0.0, multi_tile=False)
        return exps, fset, meta, info
    orig_load = data.load_field
    data.load_field = fake_load
    out = os.path.join(workdir, "run")
    try:
        m = cli.main(list(extra) + OPTIONS + ["--protocol-root", os.path.join(workdir, "proto"), "--out", out, "--method-name", "two_inst"])
    finally:
        data.load_field = orig_load
        sys.path.pop(0)
    st = torch.load(os.path.join(out, "fieldop_state.pt"), map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    flat = {k: v for k, v in st.items() if torch.is_tensor(v)}
    flat.update({f"sky{t}.{k}": v for t, s_ in st["skies"].items() for k, v in s_.items()})
    for k in sorted(flat):
        h.update(k.encode())
        h.update(flat[k].detach().cpu().contiguous().numpy().tobytes())
    return {"heldout_summary": m["heldout_summary"], "heldout_own": {k: {kk: v[kk] for kk in ("chi2", "n", "chi2_model_noise")} for k, v in m["heldout_own"].items()},
            "state_sha256": h.hexdigest(), "instrument_mean_log_throughput": m["instrument_mean_log_throughput"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        r = run(a.pkg, d)
    r["package"] = os.path.abspath(a.pkg)
    json.dump(r, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(r["heldout_summary"], default=str), r["state_sha256"])
