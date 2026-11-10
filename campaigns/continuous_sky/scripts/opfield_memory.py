#!/usr/bin/env python
"""Parameter counts, autograd memory per pixel and time per pixel of the opfield options at the REAL configuration (no data needed).

    python scripts/opfield_memory.py [--batch-small 64 --batch-big 256] [--gpu-gib 10.75] [--out FILE.json] [--options a,b,...]

What is built: an OpField with synthetic metadata that has the channel / detector / band structure of the SMACS 0723 runs (ASSUMED, edit the dict below):
NIRCam SW 0.031 arcsec/px (detectors NRCB1-4 x F090W, F150W, F200W), NIRCam LW 0.063 (NRCBLONG x F277W, F356W, F444W), WFC3/IR 0.128 (WFC3_IR x
F105W, F125W, F140W, F160W), sky 640 x 640 (size 512 at 0.04 arcsec + halo 64), K = 12 spectral coefficients, Q = 6 quadrature nodes, hidden 96, n_bank 32,
shape-hidden 64, 3 arcsec wings, --core-radius 10 --core-pitch 0.5, 190 exposures. The pixel mix of a batch follows the pixel counts of the box
(NIRCam SW 75 per cent, LW 4.6, WFC3/IR 20).

MEMORY ESTIMATE (the method, with its approximations)
 * the forward pass of the TRAINING loss is run on CPU for two small batches B1 < B2 (all three channels in the pool proportions, sky levels with
   gradient) under torch.autograd.graph.saved_tensors_hooks, which sums the bytes of the unique storages autograd keeps for backward (the "tape").
   The slope (B2 - B1) is the tape per pixel, the intercept the pixel-independent part (sky level images and their autograd tape). PyTorch on CPU and CUDA
   save the same tensors; NOT included: transient buffers of the backward pass (a factor 1.2 is applied, as in estimate_batch_memory_mb's 1.25), the CUDA
   caching allocator's fragmentation (plan for +20 per cent more), cuDNN / grid_sample workspaces, and the forward temporaries that are not saved.
 * the PEAK is also measured: one forward + backward in a fresh process, peak resident memory (VmHWM after /proc/self/clear_refs) above the post-build
   baseline, at ONE large batch (tape about 450 MiB; the difference of two small batches is dominated by allocator noise); peak / tape (>= 1, printed as
   `peak/tape`) multiplies the tape slope and intercept: it stands for the backward transients; the factor 1.2 on top stands for CUDA allocator fragmentation.
 * static GPU memory (analytic): pool and exposure tensors 41 + 64 bytes per pixel, sky + gradient + 2 Adam moments (4 x K x 640 x 640 x 4 B), CUDA context
   0.6 GiB; 1.25e7 pixels in the pool; held-out exposures and the SWarp/drizzle coadds (nb x 640 x 640) are added as 0.3 GiB.
 * time: wall time of forward + backward per pixel on 2 CPU threads (relative numbers between options are what matters; the GPU ratio follows the node counts).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opfield as O  # noqa: E402
import smacs_four_v36 as V  # noqa: E402

CHANNELS = {"NIRCAM_SW": 0.031, "NIRCAM_LW": 0.063, "WFC3_IR": 0.128}
DETS = {"NRCB1": "NIRCAM_SW", "NRCB2": "NIRCAM_SW", "NRCB3": "NIRCAM_SW", "NRCB4": "NIRCAM_SW", "NRCBLONG": "NIRCAM_LW", "WFC3_IR": "WFC3_IR"}
BANDS = {"F090W": "NIRCAM_SW", "F150W": "NIRCAM_SW", "F200W": "NIRCAM_SW", "F277W": "NIRCAM_LW", "F356W": "NIRCAM_LW", "F444W": "NIRCAM_LW",
         "F105W": "WFC3_IR", "F125W": "WFC3_IR", "F140W": "WFC3_IR", "F160W": "WFC3_IR"}
LAM = {"F090W": 0.90, "F150W": 1.50, "F200W": 2.00, "F277W": 2.77, "F356W": 3.56, "F444W": 4.44, "F105W": 1.05, "F125W": 1.25, "F140W": 1.40, "F160W": 1.60}
PIX_SHARE = {"NIRCAM_SW": 0.75, "NIRCAM_LW": 0.046, "WFC3_IR": 0.204}
BASE = ["--loss", "huber", "--core-radius", "10", "--core-pitch", "0.5", "--noise-start", "2.0"]
OPTIONS = {
    "baseline": [],
    "wing-nang72": ["--wing-nang", "72"],
    "wings-cartesian": ["--wings", "cartesian"],
    "base-per-instrument": ["--base-per", "instrument"],
    "base-per-detector": ["--base-per", "detector"],
    "base-per-detector_band": ["--base-per", "detector_band"],
    "shape-head-per-channel": ["--shape-head-per", "channel"],
    "base-table-detector_band": ["--base-table", "detector_band"],
    "base-table-detector_band-rank32": ["--base-table", "detector_band", "--mod-rank", "32"],
}


def build(flags, n_pool_exp=190, hw=640, dev=torch.device("cpu")):
    a = O.parse_args(["--out", "/tmp/unused", "--scale", "0.04", "--size", "512", "--halo-px", "64"] + BASE + list(flags))
    bands = list(BANDS)
    chan_names = list(CHANNELS)
    reg = O.OpReg(bands, [LAM[b] for b in bands], ["NIRCam" if BANDS[b].startswith("NIRCAM") else "WFC3IR" for b in bands], [BANDS[b] for b in bands], [False] * len(bands))
    dets = list(DETS)
    cells = [(d, b) for d in dets for b in bands if DETS[d] == BANDS[b]]
    rng = np.random.default_rng(0)
    exp_cell = [cells[i % len(cells)] for i in range(n_pool_exp)]
    band = torch.tensor([reg.index[b] for _, b in exp_cell])
    det = torch.tensor([dets.index(d) for d, _ in exp_cell])
    n = len(exp_cell)
    meta = dict(band=band, inst=torch.tensor([reg.inst_of_band[int(b)] for b in band]), chan=torch.tensor([reg.chan_of_band[int(b)] for b in band]), det=det,
                epoch=torch.zeros(n), ltexp=torch.zeros(n), roll=torch.tensor([[1.0, 0.0, 1.0]]).repeat(n, 1), lscale=torch.zeros(n), ro=torch.zeros(n, 2),
                J=torch.eye(2).repeat(n, 1, 1), unit=torch.full((n,), 0.1), b0=torch.zeros(n), stripe_exp=torch.zeros(n))
    Q, K = a.spec_q, a.spec_k
    A0 = torch.randn(K, hw, hw) * 0.01
    basis = V.RBFSpectralBasis(K)
    lam_nodes = torch.stack([torch.linspace(LAM[b] * 0.95, LAM[b] * 1.05, Q) for b in bands])
    w_nodes = torch.full((len(bands), Q), 1.0 / Q)
    Phi = basis(lam_nodes.flatten()).view(len(bands), Q, K).detach()
    scale_ch = [{"sky_per_det_px": CHANNELS[c] / 0.04, "arcsec_per_det_px": CHANNELS[c]} for c in reg.channels]
    Rmax = max(a.wing_arcsec / s["arcsec_per_det_px"] for s in scale_ch)
    st = O.make_stencil(a, Rmax)
    m = O.OpField(a, reg, meta, st, A0, Phi, w_nodes, lam_nodes, len(dets), scale_ch, dev, det_names=dets)
    m.set_train_rows(list(range(n)))
    return m, a, reg, st, exp_cell


def make_batch(m, reg, B, rng):
    chan_of_exp = m.m_chan.numpy()
    eids = []
    for ci, c in enumerate(reg.channels):
        pool = np.nonzero(chan_of_exp == ci)[0]
        k = int(round(B * PIX_SHARE[c]))
        eids.append(rng.choice(pool, k))
    eid = torch.tensor(np.concatenate(eids), dtype=torch.long)
    n = eid.numel()
    x0 = torch.tensor(rng.uniform(40, 600, (n, 2)), dtype=torch.float32)
    dxy = torch.tensor(rng.uniform(-1, 1, (n, 2)), dtype=torch.float32)
    return dict(eid=eid, nid=eid.clone(), x0=x0, dx=dxy[:, 0], dy=dxy[:, 1], sp=torch.ones(n) * 0.1, row=torch.randint(0, 2048, (n,)), col=torch.randint(0, 2048, (n,)), d=torch.randn(n) * 0.1)


def fwd(m, a, bt):
    lv = m.make_levels(0.0)
    out = m(bt, m.nuis, lv)
    nll, _ = O.nll_terms(bt["d"], out["pred"], bt["sp"], out["h"], out["pi"])
    S = m.band_image()
    return 2 * nll.mean() + 0.02 * S.abs().mean() + a.bank_l2 * m.bank_l2()


def measure(m, a, reg, B, rng):
    """(autograd tape MiB, seconds forward + backward, number of pixels). Tape = bytes of the UNIQUE storages that autograd saves for backward (every
    tensor saved by the forward pass of the training loss, including the sky level images, parameters and buffers they involve)."""
    bt = make_batch(m, reg, B, rng)
    m.zero_grad(set_to_none=True)
    fwd(m, a, bt).backward()                          # warm-up
    seen = {}

    def pack(t):
        st = t.untyped_storage()
        seen[(st.data_ptr(), st.nbytes())] = st.nbytes()
        return t

    m.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = fwd(m, a, bt)
    loss.backward()
    dt = time.perf_counter() - t0
    return sum(seen.values()) / 2 ** 20, dt, bt["eid"].numel()


def _status_kb(key):
    for ln in open("/proc/self/status"):
        if ln.startswith(key):
            return int(ln.split()[1])
    return 0


def child(option, B, threads):
    """One forward + backward in a FRESH process; prints the peak resident memory above the post-build baseline [MiB] and the number of pixels."""
    torch.set_num_threads(threads)
    m, a, reg, st, cells = build(OPTIONS[option])
    bt = make_batch(m, reg, B, np.random.default_rng(1))
    try:
        open("/proc/self/clear_refs", "w").write("5")
    except Exception:  # noqa: BLE001
        pass
    base = _status_kb("VmRSS")
    fwd(m, a, bt).backward()
    print(json.dumps({"peak_MiB": (_status_kb("VmHWM") - base) / 1024.0, "n": int(bt["eid"].numel())}))


def measure_peak(option, B, threads):
    import subprocess
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", option, str(B), "--threads", str(threads)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    d = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])
    return d["peak_MiB"], d["n"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", nargs=2, default=None, metavar=("OPTION", "B"))
    ap.add_argument("--batch-small", type=int, default=64)
    ap.add_argument("--batch-big", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8192, help="the batch of the real run")
    ap.add_argument("--gpu-gib", type=float, default=10.75)
    ap.add_argument("--pool-pixels", type=float, default=1.25e7)
    ap.add_argument("--options", default=",".join(OPTIONS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-peak", action="store_true", help="skip the fresh-process peak-resident-memory measurement")
    a_ = ap.parse_args(argv)
    if a_.child:
        return child(a_.child[0], int(a_.child[1]), a_.threads)
    torch.set_num_threads(a_.threads)
    rows = {}
    K, hw = 12, 640
    static_gib = (a_.pool_pixels * (41 + 64) + 4 * K * hw * hw * 4 + 10 * 1.0 * hw * hw * 4 + 0.3 * 2 ** 30 + 0.6 * 2 ** 30) / 2 ** 30
    for name in a_.options.split(","):
        flags = OPTIONS[name]
        rng = np.random.default_rng(1)
        m, a, reg, st, cells = build(flags)
        pc = m.param_counts()
        m1, t1, n1 = measure(m, a, reg, a_.batch_small, rng)
        m2, t2, n2 = measure(m, a, reg, a_.batch_big, rng)
        formula = O.estimate_batch_memory_mb(a_.batch, m.nk_max, a.spec_q, a.spec_k, hw=(hw, hw), H=a.hidden) / 1024.0
        slope_tape = (m2 - m1) / (n2 - n1)          # MiB per pixel (autograd tape only)
        inter_tape = m1 - slope_tape * n1
        if a_.no_peak:
            factor, peaks = 1.0, None
        else:
            # peak / tape at one LARGE batch (tape ~ 450 MiB): the difference of two small batches is dominated by allocator noise (~100 MiB)
            Bp = int(min(4096, max(256, 450.0 / slope_tape)))
            pk, nq = measure_peak(name, Bp, a_.threads)
            factor = max(1.0, pk / (inter_tape + slope_tape * nq))
            peaks = {"B": nq, "peak_MiB": pk, "tape_MiB": inter_tape + slope_tape * nq, "peak_over_tape": factor}
        slope, inter = slope_tape * factor, inter_tape * factor
        tslope = (t2 - t1) / (n2 - n1)
        per_pix_kib = slope * 1024
        est_gib = (inter + slope * a_.batch) / 1024.0
        total = est_gib * 1.2 + static_gib       # 1.2: allocator fragmentation / cache (not measured on CPU)
        bmax = int(max(0.0, (a_.gpu_gib - static_gib) / 1.2 - inter / 1024.0) * 1024 / slope) if slope > 0 else 0
        rows[name] = {"flags": flags, "nk_per_channel": dict(zip(reg.channels, m.nk_ch)), "stencil_points_total": len(st), "param_counts": pc,
                      "activation_KiB_per_pixel": per_pix_kib, "tape_KiB_per_pixel": slope_tape * 1024, "pixel_independent_MiB": inter, "peak_measurement": peaks, "CPU_ms_per_1000_pixels": tslope * 1e6,
                      "est_batch_activation_GiB_at_batch": est_gib, "est_total_GPU_GiB_at_batch (x1.2 fragmentation + static)": total,
                      "static_GiB": static_gib, "max_batch_in_gpu": bmax, "opfield.estimate_batch_memory_mb_GiB_at_batch": formula, "n_base_rows": m.n_rows,
                      "measured": {"B": [n1, n2], "peak_MiB": [m1, m2], "seconds": [t1, t2]}}
        print(f"{name:34s} nk {dict(zip(reg.channels, m.nk_ch))} params: kernel {pc['kernel_total (base+bank+shape_mlp)']:>8d} (base {pc['base']}, bank {pc['bank']}, mlp {pc['shape_mlp']}), "
              f"total {pc['total']}; {per_pix_kib:7.1f} KiB/px peak ({slope_tape * 1024:6.1f} tape, peak/tape {factor:.2f}; formula in opfield.py: {formula * 1024 * 1024 / a_.batch:7.1f}), fixed {inter:6.0f} MiB, {tslope * 1e6:7.2f} ms/1000 px fwd+bwd (CPU) -> est. {total:5.2f} GiB at B={a_.batch}; max batch {bmax}", flush=True)
        del m
    if a_.out:
        json.dump(rows, open(a_.out, "w"), indent=1, default=float)
    return rows


if __name__ == "__main__":
    main()
