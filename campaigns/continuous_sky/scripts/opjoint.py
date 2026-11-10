#!/usr/bin/env python3
"""opjoint.py: multi-field JOINT GLOBAL-LOCAL inversion around the field operator of opfield.py ("a gigantic global-local inverse problem inverting the sky and
instruments"). NOT "learn on field A, freeze, hope it transfers": ONE learned operator (geometry mu, PSF Psi, transmission a, detector terms, noise head) is fitted
together with a catalogue-free sky per tile and a code per exposure, over any mix of fields (LMC tiles of a plan_tiles.py plan; the SMACS field as opfield loads it:
JWST NIRCam _cal + HST WFC3/IR _flt with the a-priori WCS and opfield's own coarse shift). Transfer (leave-field-out, T5) is only the VALIDATION, not the method.
opfield.py is imported unchanged (OpField, Pool, PeakSampler, fit_nuisance, evaluate, xfit_step, c2f_sigma, write_outputs, ...); tilefit.py supplies the A13
registration (apply_external_shifts, shift_exposure_header) and the pooling helpers. See opjoint-design.md.

MODEL (tile k, exposure i, pixel p):  d = [exp(a) sum_q w_q sum_n W_n s_k(x0 + mu + J Lc (v_n - c))]_{opfield} + g + B_i + stripes_i + noise, exactly opfield's forward,
with the sky s_k of the ACTIVE tile (model.sky is swapped) and one global exposure table (meta: J, roll, band, detector, epoch, exposure time ...).

GLOBAL (one copy, one Adam, shared by every tile):  hypernetwork trunk, instrument and detector embeddings, heads for mu / Lambda / a, shape MLP, bank, base rows
(--base-per), two-scale wing tables and fractions (--two-scale), g / noise / outlier nets, per-detector pattern pd, mu-skip.
LOCAL:  the sky of each tile (12 RBF coefficients per pixel, + its cross-fit views), per-exposure code z_i, quadratic background pe and NIRCam stripes, the hierarchical
epoch / visit codes (--z-hier) and the per-tile constant background b0. An exposure that appears in several tiles has ONE z_i / pe / stripes row (one global exposure id);
its pixels, x0 and b0 are per tile. Exposures of a T5 (held-out) tile get PRIVATE ids (fresh z, pe, stripes and codes, never trained by any other tile).

CYCLE (variable projection / cross-fitting, as tilefit but with opfield's operator): pick a tile k (probability p_k, below); (a) --inner-sky K steps on s_k ONLY (every other
parameter requires_grad=False, own batch from the tile's pool or its PeakSampler, c2f and cross-fitting as opfield); (b) ONE joint step: the loss of a fresh batch (+ sky
regularisers, priors of the tile's exposures, gauges, wing prior) updates the global parameters, the nuisances and s_k (Adam objects: operator, tile sky). A visit = --visit-cycles
cycles on one tile (the swap is amortised). Learning-rate schedule, coarse-to-fine (c2f_sigma), noise-head start, sky warm-up: opfield's, on the global progress
max(cycle / --steps, elapsed / --budget-min). GPU: theta + Adam state + all per-exposure nuisances (small), the active tile's sky + Adam (+ views), its pixel pool (a random
1/chunks of it when it exceeds --gpu-pool-gb) and its PeakSampler (one per tile, rebuilt and refreshed at each visit); all else on the CPU (pools in RAM, 33 B/pixel, or --tile-store disk).
SAMPLING WEIGHTS  p_k proportional to (training pixels of tile k)^--weight-power, times --tile-weights overrides (KEY=mult, key = tile or field), the total share of a FIELD
floored at --field-floor (default 0.2; so SMACS is not starved by ten LMC tiles), then renormalised. --inst-weights NIRCam=1,WFC3/IR=2: per-instrument multipliers of the
per-pixel loss (default 1). --tile-order roundrobin | random (uniform) ignore the weights.

TESTS AND SCORES (deliverable):
  T1  held-out dithers per tile (opfield.evaluate: theta and sky frozen, per-exposure z and quadratic background fitted by Adam, background re-solved by exact WLS, stripes closed
      form; chi2 / robust on all and on S/N>5 pixels with the PIPELINE sigma; dither-cluster bootstrap) per tile, per FIELD and per (field, instrument); --score-coadds scores the
      SWarp-like and drizzle coadds of the same exposures. An exposure held out in any tile is excluded from the training of every tile.
  JOINT SMACS score  the SMACS tile scored inside the joint fit (same split / masks as the single-field opfield references: JWST-only all-three 2.23 [2.17, 2.28]; JWST + WFC3/IR
      all-three NIRCam 3.20, WFC3/IR 2.33; E2-data JWST-only 2.20), written in opfield's metrics format too (metrics.json["metrics"]).
  T5  --heldout-tiles k / --heldout-tile-ids: tiles never used for the operator. Only LOCAL parameters (new sky from the tile's drizzle coadd, private z / pe / stripes) are fitted
      (--ht-steps) with the global operator frozen, then the tile's held-out dithers are scored. --ht-control repeats it with the operator ALSO refitted on that tile alone
      from the generic start (--ht-control-cycles cycles of the same loop).
  LEAVE-FIELD-OUT  --leave-out-field NAME: train on the other fields; then on NAME refit ONLY sky, exposure codes (and its hierarchical codes) and nuisances (--lo-steps) with the
      operator frozen and score T1 there. The unchanged-ness of every global parameter is VERIFIED (metrics["leave_out"]["global_param_max_abs_delta"]). Instruments absent from the
      training fields (e.g. WFC3/IR when SMACS is left out) keep the generic start kernel, so only the NIRCam score tests what the global part learned.
  KERNEL DIAGNOSTICS / DUMPS  opfield's write_outputs (effective kernels, mu fields, kernel_diagnostics) and --dump-heldout-pixels (OUT/heldout_pixels/opfield/ for SMACS, same npz
      format, so compare_operators.py works; OUT/tile<k>/heldout_pixels/opfield/ for LMC tiles).
CHECKPOINT / RESUME / BUDGET: ckpt.pt every --ckpt-every-min (operator, optimisers, every tile sky + Adam, counters, RNG states); a re-launch resumes automatically (--no-resume to
  start over), a finished training goes straight to the scores; --budget-min is the training time (opfield / tilefit convention), --steps the maximum number of cycles.
OUT  opjoint.log, metrics.json (args, approximations J1-J14, provenance, opjoint block, opfield-format SMACS metrics), tile_metrics.json, shifts_applied.json (--shifts-from),
  holdout_selection.json, opjoint_state.pt / opjoint_best.pt (operator + nuisances; + every tile's sky in opjoint_skies.pt), cubes/<tile>_sky.fits, cubes/<tile>_drizzle.fits,
  kernels/, mu_fields.npz, heldout_pixels/ (with --dump-heldout-pixels), ckpt.pt.

EVERY APPROXIMATION / SHORTCUT (J1-J14; also in metrics.json["approximations"], opfield's S1-S37 apply unchanged to the operator):
  J1  One Jacobian J, roll, parity, scale, detector-frame b0-independent meta per exposure, taken from the tile in which the exposure has most pixels (all tiles are north-up TAN with
      the same scale, so J differs by <~1e-3); the exposure's median pipeline sigma (m_unit) likewise.
  J2  The constant background b0 is per (tile, exposure) (the exposure's median in that tile, as tilefit A2) and swapped with the tile; pe, stripes, z shared.
  J3  Gauges (mean mu = 0, flux of --ref-instrument = 0, blur) are evaluated on a random subset of --gauge-exps training exposures per joint step (all of them, in opfield's
      order, when there are <= that many: then identical to opfield); m2_ref (blur-gauge reference) is the mean over ALL training exposures at the start.
  J4  Priors on z and the stripe penalty act on the exposures of the ACTIVE tile (tilefit A4); the bank / base / wing priors and gauges are global. Rows of exposures absent from the
      current tile get zero gradient: their Adam moments keep decaying and moving them for a few steps (tilefit A11).
  J5  Tile probabilities are not proportional to pixel counts because of the field floor / overrides / weight power: the objective is sum_k p_k L_k (per-pixel mean loss of tile k),
      not the pooled-pixel mean. --inst-weights are un-normalised loss multipliers.
  J6  The inner sky solve is K Adam steps (not an exact solve); the joint step moves operator and sky together; the envelope theorem holds only approximately. Nuisances are frozen in
      the inner steps.
  J7  A tile's pool on the GPU is a random 1/chunks of its pixels (chunk c = visit mod chunks of a fixed permutation) when it exceeds --gpu-pool-gb; the importance-sampling pmf is
      then relative to the chunk (unbiased for the chunk mean, hence for the tile mean over visits). The PeakSampler is rebuilt per visit and refreshed at its start and every
      --is-refresh steps of the visit; its peaks come from the CURRENT sky of that tile (no memory of the previous visit).
  J8  Validation / early stopping uses the --n-val exposures per band of every tile that has some (SMACS: opfield's; LMC tiles: --lmc-n-val, default 0), pooled robust all + src; with
      none the last state is kept. Checkpoint selection is therefore dominated by SMACS in a joint run.
  J9  Not supported (stated, refused at start): --operator factorised (its per-exposure affine is not shared across tiles), --restart-cycles, --eval-only.
  J10 T5: private exposure ids and a fresh sky, --ht-steps Adam steps with the opfield schedule (c2f over the first --c2f-frac of the steps); the control trains the operator from the
      generic start on that tile alone for --ht-control-cycles (default: mean cycles of a training tile, >= 300) cycles with the same loop, then the same local fit is NOT repeated
      (its own sky, z, nuisances are trained jointly with the operator).
  J11 --lmc-nuis-steps (default = --nuis-steps) can shorten the held-out nuisance fit on LMC tiles (300 steps x ~100 images is hours); SMACS keeps opfield's --nuis-steps so its score
      is comparable with the references.
  J12 A tile band with no training exposure (thin F090W): its coadd is zero, sig_b is the median of the other bands, the sky init of that band is zero; the spectral prior ties it.
  J13 --shifts-from (A13) shifts are translations only; they are applied to every exposure of every field found in the JSON (SMACS names are normally absent), then opfield's
      own HST coarse shift (--coarse-shift, a grid scan against a JWST coadd) runs for SMACS as in opfield.run. A12 (tilefit's per-tile scan) is not offered.
  J14 Per-field metrics pool the per-exposure records of the tiles of the field; an exposure scored in two tiles counts twice (tile-specific sky and pixels); the bootstrap resamples
      dithers within a band over all those records.
  J15 Fields are DATA (--fields-config JSON): 'plan' fields (tiles of a plan_tiles.py plan) and 'frames' fields (one tile from image directories, as opfield loads SMACS), each with an
      "args" dictionary overriding any opfield flag for that field only (data_dir, hst_*, ra, dec, size, holdout_from, n_test, n_val, coarse_shift ...). One registry (bands,
      instruments, channels, detectors) is built from the union of all exposures; (detector, band) cells, instruments and channels are shared across fields BY NAME. Exposure ids are
      file names: the same name in two fields is refused. Flags that act on the model (hidden, core radius, stencil, two-scale ...) are global; the target grid scale --scale is
      global (all tiles); the detector pixel scale of an instrument enters only through the Jacobian J (instruments of different pixel scale are supported; tested with 1 and 2 target
      px per detector px). The halo / size of a frames field is its own. The stencil extent in detector pixels is set by the channel with the smallest detector pixel.
  J16 Leave-field-out (--holdout-field NAME): (i) LOCAL-ONLY (the deliverable): the field's tiles get a fresh sky (tile drizzle), fresh codes / quadratic backgrounds / stripes, and
      only these are fitted (--lo-steps) with every global parameter frozen (verified bit-identical, reported); an instrument or (channel) cell absent from the training fields
      keeps its generic start kernel, so e.g. WFC3/IR is NOT transferable when SMACS is the only WFC3/IR field. (ii) with --lo-ft-cycles N: a NEW fit over ALL tiles, the left-out field now
      training, started from (i)'s operator / nuisances / skies, N cycles of the normal loop at --lo-ft-lr-mult x learning rates (fresh Adam moments, no coarse-to-fine restart, noise head
      on from the start when the run uses it, no cross-fitting). Both are reported (T1 and T1_finetune); the fine-tune moves the shared operator, so it is not a transfer score.
  J17 Checkpoints / best-validation files are per fit (ckpt.pt, opjoint_best.pt for the main fit; ckpt_<tag>.pt for the T5 control and the fine-tune), so a control never overwrites
      the main run's resume point.
  J18 LMC T1 selection = tilefit's rule (V.select_exposures, n_test = --n-test-tile / field "n_test_tile") but with the bands PRESENT in the tile (tilefit passes all registry bands: a band
      absent from a tile yields a warning in tilefit, nothing here). v36 and v38 select_exposures are textually identical. SMACS: opfield's rule and holdout file (holdout_q5_cal.json).
  J19 Registration: a field's "shifts_from" (A13) shifts the exposures found in that file in that field's tiles; --shifts-from applies to every tile without its own entry (frames
      fields included). A frames field then runs opfield's own coarse shift (field arg coarse_shift) after A13. Several files give several summaries (shifts_applied_<n>.json).
  J20 Equivalence with opfield.py: ONE field, ONE tile, --inner-sky 0, --visit-cycles >= steps, no cross-fitting, <= --gauge-exps training exposures, reproduces opfield.train BIT FOR
      BIT on the CPU (same OpField init, pool order, generator draws, learning-rate schedule, gradient terms; tests/test_opjoint.py compares every parameter after 25 steps: max
      difference 0). Not bit-identical: (a) with the defaults (--inner-sky 2, --visit-cycles 25) by construction (extra sky steps; pool re-uploaded and PeakSampler rebuilt every visit);
      (b) on a GPU, where the scatter-adds of both programs are non-deterministic; (c) with --importance-sampling the refresh schedule coincides only for visit >= steps (not
      checked bit for bit); (d) with --xfit-reg (folds are over all training exposures, views per tile; Adam objects differ). The evaluation code is opfield's own.
  Speed / memory numbers in the logs are estimates unless stated "measured".

Usage: opjoint.py --out DIR --fields-config fields_p1j.json [--holdout-field SMACS --lo-ft-cycles 200] [--heldout-tiles K --ht-control] [--shifts-from register_xcorr.json]
  [opfield flags: --args-from, --psi-param log, --importance-sampling, --two-scale, --loss huber, --lambda-gauge jacobian, --halo-px, --budget-min ...]
  (legacy: --fields LMC,SMACS --plan PLAN.json with opfield's own data flags describe the same two fields without a config)
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opfield as O  # noqa: E402
import tilefit as TF  # noqa: E402

V = O.V
TF.V = V    # tilefit's helpers (A13 shifts, pooling) run on the v36 engine that opfield uses
log = O.log

STATE_FORMAT = "opjoint-state"
REFERENCES = {"SMACS JWST-only all-three (opfield_v5_jwst_all3) NIRCam S/N>5": "2.23 [2.17, 2.28]",
              "SMACS JWST+WFC3/IR all-three (opfield_v5_wfc3ir_all3) NIRCam S/N>5": 3.20, "SMACS JWST+WFC3/IR all-three WFC3/IR S/N>5": 2.33,
              "SMACS E2-data JWST-only S/N>5": "2.20 [2.14, 2.26]"}

APPROXIMATIONS = O.APPROXIMATIONS + [
    "J1 one Jacobian / roll / scale / unit per exposure from the tile where it has most pixels",
    "J2 b0 per (tile, exposure), swapped with the tile; pe, stripes, z shared",
    "J3 gauges on a random subset of --gauge-exps training exposures per joint step (all of them, in opfield's order, when <= that); m2_ref over all",
    "J4 z prior and stripe penalty restricted to the active tile's exposures; absent rows get zero gradient (Adam moments decay)",
    "J5 tile probabilities have a field floor / overrides: objective = sum_k p_k L_k, not the pooled pixel mean; --inst-weights un-normalised",
    "J6 inner sky solve = K Adam steps (not exact VarPro); nuisances frozen in the inner steps",
    "J7 GPU pool = random 1/chunks of the tile when it exceeds --gpu-pool-gb; PeakSampler rebuilt and refreshed per visit from the current sky",
    "J8 validation from the --n-val exposures of the tiles that have some (SMACS; LMC --lmc-n-val default 0): checkpoint selection dominated by SMACS",
    "J9 not supported: --operator factorised, --restart-cycles, --eval-only",
    "J10 T5: private ids, fresh sky, --ht-steps; control = operator refitted from the generic start on the tile alone for --ht-control-cycles cycles",
    "J11 --lmc-nuis-steps may shorten the held-out nuisance fit of LMC tiles; SMACS keeps --nuis-steps",
    "J12 a tile band with no training exposure: zero coadd, sig_b = median of the other bands, zero sky init for it",
    "J13 --shifts-from (A13) translations only, applied to every exposure found in the JSON; SMACS then keeps opfield's HST coarse shift",
    "J14 per-field metrics pool the records of the field's tiles; an exposure scored in two tiles counts twice",
    "J15 fields = JSON config (plan / frames, per-field opfield args); registry, cells and instruments shared across fields by name; global --scale; unique exposure file names",
    "J16 leave-field-out: local-only refit with all global parameters frozen (verified); optional second stage = fine-tune of everything from that state (--lo-ft-cycles), both reported",
    "J17 per-fit checkpoint files (control / fine-tune fits do not touch the main ckpt)",
    "J18 LMC T1 selection: tilefit's rule with the bands present in the tile",
    "J19 per-field A13 shifts; --shifts-from for tiles without their own entry; opfield coarse shift of frames fields after A13",
    "J20 single field + tile + K=0 + visit>=steps: bit-identical to opfield.train on CPU (tested); defaults and GPU are not",
]
APPROX_A13 = TF.APPROX_A13


# ====================================================================================================================== small helpers
def lr_factor(frac, warm=0.03):
    """opfield.train.set_lr factor (warm-up + cosine), without the sky warm-up gate."""
    return max(frac, 1e-3) / warm if frac < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - warm) / (1 - warm))))


def sha16(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except Exception:  # noqa: BLE001
        return "?"


def move_exps(exps, device):
    for e in exps:
        for k, v in list(e.items()):
            if torch.is_tensor(v):
                e[k] = v.to(device)


def gb(nbytes):
    return nbytes / 2 ** 30


def gpu_mem():
    if torch.cuda.is_available():
        return {"allocated_GiB": round(gb(torch.cuda.memory_allocated()), 3), "max_allocated_GiB": round(gb(torch.cuda.max_memory_allocated()), 3),
                "reserved_GiB": round(gb(torch.cuda.memory_reserved()), 3)}
    return {}


def parse_kv(txt, what):
    out = {}
    for item in [x for x in str(txt or "").split(",") if x.strip()]:
        if "=" not in item:
            raise SystemExit(f"{what}: expected KEY=VALUE, got {item!r}")
        k, v = item.rsplit("=", 1)
        out[k.strip()] = float(v)
    return out


class ScoredExps:
    """eid -> exposure dict (CPU, full tensors) that materialises ONE exposure on the device at a time (what opfield.evaluate needs: exps[i])."""

    def __init__(self, d, dev):
        self.d, self.dev, self._c = d, dev, (None, None)

    def __getitem__(self, i):
        if self._c[0] == i:
            return self._c[1]
        e = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in self.d[i].items()}
        for k in ("row", "col"):
            e[k] = e[k].long()
        self._c = (i, e)
        return e

    def __len__(self):
        return len(self.d)


class GPool(O.Pool):
    """opfield.Pool whose columns come from a compact CPU store (subset idx) instead of exposure dicts. draw / get / bright_idx are opfield's."""

    def __init__(self, store, idx, dev):
        sel = None if idx is None else idx
        g = (lambda v: v if sel is None else v[sel])
        self.eid = g(store["eid"]).to(dev).long()
        self.x0 = g(store["x0"]).to(dev)
        self.dx, self.dy, self.d, self.sp = (g(store[k]).to(dev) for k in ("dx", "dy", "d", "sp"))
        self.row, self.col = g(store["row"]).to(dev).to(torch.int32), g(store["col"]).to(dev).to(torch.int32)
        self.bright = g(store["bright"]).to(dev)
        self.N = int(self.d.numel())
        self.bright_idx = torch.nonzero(self.bright).squeeze(1)
        self.device = dev

    def nbytes(self):
        return sum(int(t.numel() * t.element_size()) for t in (self.eid, self.x0, self.dx, self.dy, self.d, self.sp, self.row, self.col, self.bright))


# ====================================================================================================================== tiles and the global exposure table
class Tile:
    """One sky: key, field, role ('train' | 'heldout' (T5) | 'leftout'), header, exposure dicts (CPU) and, after prepare(), the compact pixel store."""

    def __init__(self, key, field, thead, exps, size_data, halo, role="train", sel=None, tile_id=None, coarse=False, kind=None, fa=None, spec=None):
        self.key, self.field, self.thead, self.exps, self.role = key, field, thead, exps, role
        self.kind = kind or ("frames" if coarse else "plan")     # 'frames' (SMACS-like: one tile from image directories) | 'plan' (tiles of a plan_tiles.py plan)
        self.fa = fa                                              # opfield args of this tile's field (field "args" overrides applied); None = the global args
        self.spec = spec or {}
        self.opfield_metrics = False                              # the (first) frames field also reports opfield's metrics.json block
        self.size_data, self.halo, self.sel, self.id, self.coarse = size_data, halo, sel, (key if tile_id is None else tile_id), coarse
        self.eid = {}
        self.store = None
        self.n_pix = 0
        self.visits = 0
        self.cycles = 0
        self.skipped = []

    def get_store(self):
        if isinstance(self.store, (str, Path)):
            return torch.load(self.store, map_location="cpu", weights_only=False)
        return self.store


class GlobalTable:
    """One row per global exposure id. Key = file name (private ids of a T5 tile: name@tile). meta = opfield.build_meta over one representative per id."""

    def __init__(self, tiles, reg, dev):
        self.eid_of, rep, self.keys = {}, {}, []
        owner = {}
        for t in tiles:
            t.eid = {}
            for e in t.exps:
                if owner.setdefault(e["name"], t.field) != t.field:
                    raise SystemExit(f"exposure file name {e['name']!r} occurs in two fields ({owner[e['name']]}, {t.field}); exposure ids are file names")
                key = e["name"] if t.role != "heldout" else f"{e['name']}@{t.key}"
                if key not in self.eid_of:
                    self.eid_of[key] = len(self.keys)
                    self.keys.append(key)
                eid = self.eid_of[key]
                t.eid[e["name"]] = eid
                n = int(e["d"].numel())
                if eid not in rep or n > rep[eid][0]:
                    rep[eid] = (n, e, t.key)
        self.n = len(self.keys)
        self.rep = [rep[i][1] for i in range(self.n)]
        self.rep_tile = [rep[i][2] for i in range(self.n)]
        self.meta, self.mjd, self.tex = O.build_meta(self.rep, reg, dev)
        priv = [i for i, k in enumerate(self.keys) if "@" in k]
        if priv:     # private ids: fresh epoch / visit rows (never shared with a trained exposure), only relevant for --z-hier
            ep, vis = self.meta["ep_id"].clone(), self.meta["vis_id"].clone()
            new_e, new_v = {}, {}
            for i in priv:
                tk = self.rep_tile[i]
                ke, kv = (int(ep[i]), tk), (O.visit_key(self.rep[i]["name"]), tk)
                ep[i] = int(self.meta["ep_id"].max()) + 1 + new_e.setdefault(ke, len(new_e))
                vis[i] = int(self.meta["vis_id"].max()) + 1 + new_v.setdefault(kv, len(new_v))
            self.meta["ep_id"], self.meta["vis_id"] = ep, vis
        self.dxdy = {}


def sample_dxdy(gt, n=4096, seed=4321):
    """A fixed random sample of detector coordinates per global exposure (kernel diagnostics need only these; avoids keeping every pixel on the device)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for e in gt.rep:
        m = int(e["dx"].numel())
        s = torch.randperm(m, generator=g)[:min(n, m)]
        out.append((e["dx"].cpu()[s], e["dy"].cpu()[s]))
    return out


# ====================================================================================================================== loading (real data)
def field_args(a, spec):
    """opfield args of one field: a copy of the global args with the field's "args" dictionary applied (keys with - or _; unknown keys are an error)."""
    fa = copy.copy(a)
    for k, v in (spec.get("args") or {}).items():
        k = k.replace("-", "_")
        if not hasattr(fa, k):
            raise SystemExit(f"field {spec.get('name')}: unknown opfield argument {k!r}")
        setattr(fa, k, v)
    return fa


def load_frames_field(spec, a, aj, first_frames):
    """A 'frames' field (SMACS-like): one tile, exposures of image directories (JWST _cal in data_dir, HST _flt via hst_dir ...) exactly as opfield.main loads them, CPU tensors.
    Everything opfield takes on its command line (ra, dec, size, scale, data_dir, hst_*, holdout_from, n_test, n_val, coarse_shift ...) can be set per field in spec["args"]."""
    name = spec["name"]
    fa = field_args(a, spec)
    a_s = copy.copy(fa)
    a_s.size_data = fa.size
    if fa.halo_px > 0:
        a_s.size = fa.size + 2 * fa.halo_px
    thead = V.target_header(fa.ra, fa.dec, a_s.size, fa.scale)
    pad = int(round(24 * 0.04 / fa.scale)) - fa.halo_px
    V.args_global = a_s
    V.CALIBRATE_NOISE = False
    exps = V.load_all_exposures(a_s, thead, torch.device("cpu"), pad)
    log(f"{name}: loaded {len(exps)} exposures, {sum(e['d'].numel() for e in exps) / 1e6:.2f} M pixels")
    if not exps:
        raise SystemExit(f"{name}: no exposures loaded")
    n_test_arg = getattr(fa, "n_test", None)
    sel = dict(n_test=1 if n_test_arg is None else n_test_arg, n_val=fa.n_val, holdout_names=(V.read_holdout_names(fa.holdout_from) if fa.holdout_from else None),
               no_holdout=fa.no_holdout, fill_missing_bands=n_test_arg is not None)
    t = Tile(name, name, thead, exps, fa.size, fa.halo_px, sel=sel, coarse=True, kind="frames", fa=fa, spec=spec)
    t.opfield_metrics = bool(spec.get("opfield_metrics", first_frames))
    return [t]


def load_plan_field(spec, a, aj, out, only_field):
    """A 'plan' field (LMC-like): the tiles of a plan_tiles.py JSON (tilefit.load_tile_exposures with the v36 loader), CPU tensors. Tile key NAME:id.
    spec keys (default = the same-named command-line flags): plan, tiles, heldout_tile_ids, heldout_tiles, tile_use, n_test_tile, n_val, nuis_steps, args."""
    name = spec["name"]
    g = lambda k, d: spec.get(k, d)  # noqa: E731
    fa = field_args(a, spec)
    plan = json.load(open(g("plan", aj.plan)))
    recs = plan["tiles"]
    ids = g("tiles", aj.tiles)
    if ids:
        keep = {int(x) for x in (ids.split(",") if isinstance(ids, str) else ids)}
        recs = [r for r in recs if r["id"] in keep]
    if not recs:
        raise SystemExit(f"{name}: no tiles selected")
    hid = g("heldout_tile_ids", aj.heldout_tile_ids)
    nho = int(g("heldout_tiles", aj.heldout_tiles))
    if hid:
        ho = {int(x) for x in (hid.split(",") if isinstance(hid, str) else hid)}
    else:
        ho = {r["id"] for r in recs[len(recs) - nho:]} if nho > 0 else set()
    if len(ho) >= len(recs) and only_field:
        raise SystemExit(f"{name}: every tile is held out")
    tile_px = int(round(plan["params"]["size_arcsec"] / fa.scale))
    size = tile_px + 2 * fa.halo_px
    pad = int(round(24 * 0.04 / fa.scale)) - fa.halo_px
    links = Path(out) / "tile_links"
    tile_use = g("tile_use", aj.tile_use)
    n_test_tile = int(g("n_test_tile", aj.n_test_tile))
    tiles = []
    for r in recs:
        thead = V.target_header(r["ra"], r["dec"], size, fa.scale)
        if tile_use == "covering":
            files = sorted({e["path"] for b in plan["bands"] for e in r["exposures"][b] if e["covering"]})
        else:
            files = list(r["files"])
        d = links / f"{name}_tile{r['id']}"
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            f.unlink()
        for f in files:
            os.symlink(f, d / os.path.basename(f))
        V.args_global = fa
        exps = V.load_exposures(str(d), thead, torch.device("cpu"), pad=pad)
        log(f"{name} tile {r['id']}: {len(exps)} of {len(files)} files loaded")
        sel = dict(n_test=max(n_test_tile, 1), n_val=int(g("n_val", aj.lmc_n_val)), holdout_names=None, no_holdout=n_test_tile == 0, fill_missing_bands=False)
        t = Tile(f"{name}:{r['id']}", name, thead, exps, tile_px, fa.halo_px, role=("heldout" if r["id"] in ho else "train"), sel=sel, tile_id=r["id"], kind="plan", fa=fa, spec=spec)
        tiles.append(t)
    return tiles


def load_fields(a, aj, out):
    """All tiles of all fields of aj.field_specs (the JSON fields config, or the legacy --fields/--plan flags turned into the same structure)."""
    specs = aj.field_specs
    tiles, first_frames = [], True
    for sp in specs:
        if sp["type"] == "frames":
            tiles += load_frames_field(sp, a, aj, first_frames)
            first_frames = False
        else:
            tiles += load_plan_field(sp, a, aj, out, only_field=len(specs) == 1)
    return tiles


def load_fields_config(path):
    """The fields config: {"fields": [{"name": "LMC", "type": "plan", "plan": "...", "tiles": [..], "heldout_tile_ids": [..], "shifts_from": "...", "n_test_tile": 1, "args": {..}},
    {"name": "SMACS", "type": "frames", "args": {"data_dir": "...", "hst_dir": "...", "ra": .., "dec": .., "holdout_from": "...", "coarse_shift": 1.0}}]}.
    A new field or instrument is a new entry (and its image directory), not code. Field names are free (they key the tile names, the metrics and --holdout-field)."""
    js = json.load(open(path))
    specs = js["fields"] if isinstance(js, dict) else js
    names = [s_["name"] for s_ in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate field names {names}")
    for s_ in specs:
        if s_.get("type") not in ("plan", "frames"):
            raise SystemExit(f"{path}: field {s_.get('name')}: type must be 'plan' or 'frames'")
        if s_["type"] == "plan" and "plan" not in s_:
            raise SystemExit(f"{path}: plan field {s_['name']} needs a 'plan' path")
    return specs


# ====================================================================================================================== assembling: registry, selection, registration, tables, tiles
def select_holdouts(tiles, reg, out=None):
    """V.select_exposures per tile (its own rule), then GLOBAL exclusion: a name held out / validation in any tile is that in every tile (tilefit A9)."""
    names_h, names_v, report = set(), set(), {}
    for t in tiles:
        if t.sel is None:     # flags preset (synthetic tests)
            names_h |= {e["name"] for e in t.exps if e.get("holdout")}
            names_v |= {e["name"] for e in t.exps if e.get("val")}
            continue
        bands = [b for b in reg.bands if any(e["band"] == b for e in t.exps)]
        s = V.select_exposures(t.exps, n_test=t.sel["n_test"], n_val=t.sel["n_val"], holdout_names=t.sel["holdout_names"], no_holdout=t.sel["no_holdout"], bands=bands,
                               fill_missing_bands=t.sel["fill_missing_bands"])
        for w in s["warnings"]:
            log(f"{t.key} selection warning: {w}")
        names_h |= set(s["holdout"])
        names_v |= set(s["validation"])
        report[t.key] = {"holdout": s["holdout"], "validation": s["validation"], "warnings": s["warnings"]}
    for t in tiles:
        for e in t.exps:
            e["holdout"] = e["name"] in names_h
            e["val"] = (e["name"] in names_v) and not e["holdout"]
    if out is not None:
        json.dump(report, open(Path(out) / "holdout_selection.json", "w"), indent=1)
    return names_h, names_v


def assemble(tiles, a, aj, dev, out, reg=None, bandpass=None, classical_fn=None, stage_log=True):
    """Everything between 'tiles loaded' and 'ready to fit': registry, detector list, hold-outs, A13 shifts, Jacobians, SMACS coarse shifts, the global exposure table,
    bandpasses, stencil, per-tile coadds / sky init / compact pools. Returns a SimpleNamespace ctx. `reg` / `bandpass` / `classical_fn` are given by the synthetic tests."""
    t_all = time.perf_counter()
    all_exps = [e for t in tiles for e in t.exps]
    if reg is None:
        V.set_registry(V.registry_from_exposures(all_exps, psf_n=31))
        reg = O.OpReg.from_v36(V.REG)
    log("bands: " + ", ".join(f"{b} [{reg.instrument[k]} {reg.channel[k]} {reg.lam_um[k]:.3f}um]" for k, b in enumerate(reg.bands)))
    V.LAM_LO, V.LAM_HI = a.lam_lo, a.lam_hi
    dets = sorted({e["detname"] for e in all_exps})
    for e in all_exps:
        e["det"] = dets.index(e["detname"])
    names_h, names_v = select_holdouts(tiles, reg, out)
    log(f"held-out detector images (global): {len(names_h)}; validation: {len(names_v)}")
    for t in tiles:
        for e in t.exps:
            e["tile"] = t.key
    for t in tiles:
        for e in t.exps:
            e["jac"] = V.estimate_jacobian(e)
    groups = {}      # A13 (tilefit): a tile's shifts file = its field's "shifts_from", else --shifts-from; exposures found in the JSON are shifted (tx, ty, header CRVAL) in every such tile
    for t in tiles:
        f_ = t.spec.get("shifts_from", aj.shifts_from)
        if f_:
            groups.setdefault((f_, bool(t.spec.get("shifts_from_by_tile", aj.shifts_from_by_tile))), []).append(t)
    shifts = []
    for n_, ((f_, by_), ts) in enumerate(groups.items()):
        ad = [SimpleNamespace(id=t.id, thead=t.thead, exps=t.exps, key=t.key) for t in ts]
        shifts.append(TF.apply_external_shifts(ad, f_, by_tile=by_, out=(out if len(groups) == 1 else None)))
        if len(groups) > 1:
            json.dump(shifts[-1], open(Path(out) / f"shifts_applied_{n_}.json", "w"), indent=1)
    shifts = shifts[0] if len(shifts) == 1 else (shifts or None)
    for t in tiles:     # frames fields (SMACS): opfield.run's registration, unchanged (HST grid scan, else the JWST scan), with the field's own args
        if not t.coarse:
            continue
        fa = t.fa or a
        V.args_global = fa
        if fa.coarse_shift > 0 or fa.coarse_shift_jwst > 0:
            move_exps(t.exps, dev)
            if fa.coarse_shift > 0 and any(reg.instrument[reg.index[e["band"]]] != "NIRCam" for e in t.exps):
                V.coarse_register_hst(t.exps, t.thead, dev, fa)
            elif fa.coarse_shift_jwst > 0:
                V.coarse_register_all(t.exps, t.thead, dev, fa)
            move_exps(t.exps, torch.device("cpu"))
    V.args_global = a
    gt = GlobalTable(tiles, reg, dev)
    log(f"global exposure table: {gt.n} exposures ({sum(1 for k in gt.keys if '@' in k)} private ids of held-out tiles)")
    if bandpass is None:
        lam_nodes, w_nodes, _ = V.load_bandpasses(reg.bands, a.spec_q, a.svo_cache)
    else:
        lam_nodes, w_nodes = bandpass
    lam_nodes, w_nodes = lam_nodes.to(dev), w_nodes.to(dev)
    basis = V.RBFSpectralBasis(a.spec_k, learned=False).to(dev)
    Phi_nodes = basis(lam_nodes.flatten()).view(len(reg.bands), -1, a.spec_k).detach()
    scale_ch = O.channel_scales(gt.rep, reg, gt.meta["band"], a.scale)
    Rmax = max((a.two_scale if a.two_scale > 0 else a.wing_arcsec) / s["arcsec_per_det_px"] for s in scale_ch)
    stencil = O.make_stencil(a, Rmax)
    dxdy = sample_dxdy(gt)
    gexps = [dict(name=e["name"], band=e["band"], detname=e["detname"], det=e["det"], jac=e["jac"], path=e.get("path"), dx=dxdy[i][0], dy=dxdy[i][1], bkg0=e["bkg0"],
                  instrument=reg.instrument[reg.index[e["band"]]]) for i, e in enumerate(gt.rep)]
    ctx = SimpleNamespace(a=a, aj=aj, dev=dev, reg=reg, dets=dets, gt=gt, lam_nodes=lam_nodes, w_nodes=w_nodes, basis=basis, Phi_nodes=Phi_nodes, scale_ch=scale_ch, stencil=stencil,
                          band_index=dict(reg.index), tiles=tiles, classical_fn=classical_fn, shifts=shifts, names_h=names_h, names_v=names_v, gexps=gexps, out=Path(out))
    for t in tiles:
        prepare_tile(t, ctx)
    ctx.seconds_assemble = time.perf_counter() - t_all
    return ctx


def prepare_tile(t, cx):
    """Coadds, masks, noise, sky init, apertures, compact pixel store of ONE tile; the big per-exposure tensors of training exposures are released afterwards."""
    a, reg, dev, aj = cx.a, cx.reg, cx.dev, cx.aj
    exps, gt = t.exps, cx.gt
    t0 = time.perf_counter()
    cfn = cx.classical_fn or V.classical
    H, W = int(t.thead["NAXIS2"]), int(t.thead["NAXIS1"])
    present = [b for b in reg.bands if any(e["band"] == b and not e["holdout"] and not e.get("val") for e in exps)]
    if not present:
        raise SystemExit(f"tile {t.key}: no training exposure")
    kw = {} if len(present) == len(reg.bands) else {"bands": present}
    swarp = cfn(exps, t.thead, "lanczos3", 1.0, dev, **kw)
    driz = cfn(exps, t.thead, "drizzle_square", 0.8, dev, **kw)
    cover = dict(getattr(cfn, "last_cover", {}))
    if kw:     # J12: bands with no training exposure are zero images
        def expand(p):
            full = torch.zeros(len(reg.bands), H, W, device=p.device, dtype=p.dtype)
            for j, b in enumerate(present):
                full[reg.index[b]] = p[j]
            return full
        swarp, driz = expand(swarp), expand(driz)
    stats = [V.band_median_sigma(c, cover.get(b)) for b, c in zip(reg.bands, swarp)]
    sig_b = torch.stack([s_ for _, s_ in stats]).clamp(min=1e-6)
    if kw:
        med = sig_b[[reg.index[b] for b in present]].median()
        for b in reg.bands:
            if b not in present:
                sig_b[reg.index[b]] = med
    masks = []
    for (m_, _), b, c, sb in zip(stats, reg.bands, swarp, sig_b):
        sm_ = (c - m_) > 5 * sb
        if b not in present:
            sm_ = torch.zeros_like(sm_)
        if cover.get(b) is not None:
            sm_ = sm_ & cover[b]
        masks.append(sm_)
    src_masks = torch.stack(masks)
    if t.halo > 0:
        inner = torch.zeros_like(src_masks[0])
        inner[t.halo:t.halo + t.size_data, t.halo:t.halo + t.size_data] = True
        src_masks = src_masks & inner
    lam_nodes, w_nodes, basis = cx.lam_nodes, cx.w_nodes, cx.basis
    A0 = V.spectral_init(driz, lam_nodes, w_nodes, basis)
    if a.sky_c2f > 0:
        A0 = O.gauss_blur_img(A0, a.sky_c2f_arcsec / a.scale)
    det_bands = [k for k, ch in enumerate(reg.channel) if ch == "NIRCAM_SW"] or sorted(range(len(reg.bands)), key=lambda k: reg.lam_um[k])[:3]
    t.apertures = None
    if getattr(aj, "apertures", True):
        try:
            t.apertures = tuple(x.cpu() if torch.is_tensor(x) else x for x in O.build_apertures(swarp, sig_b, a, reg, det_bands))
            log(f"tile {t.key}: aperture catalogue {t.apertures[1].shape[0]} isolated sources")
        except Exception as exc:  # noqa: BLE001
            log(f"tile {t.key}: apertures skipped ({exc!r})")
    t.sig_b, t.src_masks, t.A0 = sig_b.cpu(), src_masks.cpu(), A0.detach().cpu()
    t.swarp, t.driz = swarp.cpu(), driz.cpu()
    t.sig_mean = float(sig_b.mean())
    # pixel store (opfield.Pool on the CPU, then compacted: 33 B / pixel)
    t.train_eids = sorted(t.eid[e["name"]] for e in exps if not e["holdout"] and not e.get("val"))
    t.hold_eids = sorted(t.eid[e["name"]] for e in exps if e["holdout"])
    t.val_eids = sorted(t.eid[e["name"]] for e in exps if e.get("val"))
    by_eid = {t.eid[e["name"]]: e for e in exps}
    cpu = torch.device("cpu")
    pool = O.Pool(by_eid, t.train_eids, t.src_masks, cx.band_index, cpu)
    store = dict(eid=pool.eid.to(torch.int32), x0=pool.x0, dx=pool.dx, dy=pool.dy, d=pool.d, sp=pool.sp, row=pool.row.to(torch.int16), col=pool.col.to(torch.int16), bright=pool.bright)
    t.n_pix = pool.N
    t.n_bright = int(pool.bright_idx.numel())
    gpool_bytes = 50.0 * t.n_pix
    t.chunks = max(1, int(math.ceil(gpool_bytes / (aj.gpu_pool_gb * 2 ** 30)))) if aj.pool_chunks <= 0 else int(aj.pool_chunks)
    if t.chunks > 1:
        store["perm"] = torch.randperm(pool.N, generator=torch.Generator().manual_seed(1000 + abs(hash(t.key)) % 1000)).to(torch.int32)
    del pool
    if aj.tile_store == "disk":
        d = Path(cx.out) / "tile_store"
        d.mkdir(exist_ok=True)
        p = d / (t.key.replace(":", "_") + ".pt")
        torch.save(store, p)
        t.store = str(p)
    else:
        t.store = store
    # scored exposures keep their tensors (CPU); everything else is released
    t.scored = {}
    for e in exps:
        eid = t.eid[e["name"]]
        if e["holdout"] or e.get("val"):
            for k in ("row", "col"):
                e[k] = e[k].long()
            t.scored[eid] = e
    t.b0_vec = gt.meta["b0"].detach().cpu().clone()
    for e in exps:
        t.b0_vec[t.eid[e["name"]]] = float(e["bkg0"])
    t.names_by_eid = {t.eid[e["name"]]: e["name"] for e in exps}
    t.exp_info = {t.eid[e["name"]]: dict(name=e["name"], band=e["band"], det=e["detname"], npix=int(e["d"].numel()), holdout=bool(e["holdout"]), val=bool(e.get("val"))) for e in exps}
    t.exps = [e for e in exps if e["holdout"] or e.get("val")]
    for e in exps:
        if not (e["holdout"] or e.get("val")):
            for k in ("tx", "ty", "d", "ivar", "dx", "dy", "row", "col"):
                e[k] = None
    t.n_train_exp = len(t.train_eids)
    log(f"tile {t.key} [{t.role}]: {t.n_train_exp} train / {len(t.hold_eids)} held-out / {len(t.val_eids)} validation exposures, {t.n_pix / 1e6:.2f} M training pixels "
        f"({t.n_bright / 1e6:.2f} M bright), pool chunks {t.chunks} (~{gb(50.0 * t.n_pix / t.chunks):.2f} GiB per chunk), prepared in {time.perf_counter() - t0:.0f} s")


# ====================================================================================================================== sky state of a tile
class SkyState:
    """The local sky of a tile: Parameter (+ cross-fit views) and its Adam. Lives on the CPU unless the tile is active."""

    def __init__(self, tile, a, K_x, lr_sky):
        self.sky = nn.Parameter(tile.A0.clone())
        self.views = [nn.Parameter(tile.A0.clone()) for _ in range(K_x)] if K_x >= 2 else []
        self.lr = lr_sky
        groups = [{"params": [self.sky], "lr": lr_sky}] + ([{"params": self.views, "lr": lr_sky}] if self.views else [])
        self.opt = torch.optim.Adam(groups)
        self.base_lr = [g["lr"] for g in self.opt.param_groups]
        self.switched = False

    def to(self, dev):
        for p in [self.sky] + self.views:
            p.data = p.data.to(dev)
            if p.grad is not None:
                p.grad = None
            for st in self.opt.state.get(p, {}).values() if p in self.opt.state else []:
                pass
        for st in self.opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v) and v.dim() > 0:
                    st[k] = v.to(dev)
        return self


class Both:
    """Minimal optimiser facade for opfield.xfit_step (it calls zero_grad and step on one optimiser)."""

    def __init__(self, *opts):
        self.opts = opts

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


@contextlib.contextmanager
def only_trainable(model, keep):
    """requires_grad False for every model parameter not in `keep` (a list of Parameters); restored afterwards."""
    ids = {id(p) for p in keep}
    req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in req:
        if id(p) not in ids:
            p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in req:
            p.requires_grad_(r)


# ====================================================================================================================== tile probabilities
def tile_probs(tiles, aj):
    """Sampling probability of each training tile (docstring: SAMPLING WEIGHTS). Returns {key: p}."""
    w = {t.key: float(max(t.n_pix, 1)) ** aj.weight_power for t in tiles}
    for k, m in parse_kv(aj.tile_weights, "--tile-weights").items():
        hit = [t.key for t in tiles if t.key == k or t.field == k]
        if not hit:
            raise SystemExit(f"--tile-weights: no tile or field {k!r}")
        for h in hit:
            w[h] *= m
    fields = list(dict.fromkeys(t.field for t in tiles))
    tot = sum(w.values())
    share = {f: sum(w[t.key] for t in tiles if t.field == f) / tot for f in fields}
    if len(fields) > 1 and aj.field_floor > 0:
        fl = min(aj.field_floor, 1.0 / len(fields))
        for _ in range(len(fields)):
            low = [f for f in fields if share[f] < fl - 1e-12]
            if not low:
                break
            rest = [f for f in fields if f not in low]
            rem = 1.0 - fl * len(low)
            srest = sum(share[f] for f in rest)
            for f in low:
                share[f] = fl
            for f in rest:
                share[f] = share[f] * rem / srest if srest > 0 else rem / len(rest)
    p = {}
    for t in tiles:
        fw = sum(w[x.key] for x in tiles if x.field == t.field)
        p[t.key] = share[t.field] * w[t.key] / fw
    return p


# ====================================================================================================================== the joint fit
class Fit:
    """One OpField on the global exposure table + the sky state of every tile. `tiles` = the tiles this fit knows (training tiles are those with role 'train')."""

    def __init__(self, ctx, tiles, tag="main", train_tiles=None):
        a, aj, dev, reg, gt = ctx.a, ctx.aj, ctx.dev, ctx.reg, ctx.gt
        self.cx, self.a, self.aj, self.dev, self.reg, self.gt, self.tag = ctx, a, aj, dev, reg, gt, tag
        self.sfx = "" if tag == "main" else "_" + "".join(c if c.isalnum() else "-" for c in tag)    # checkpoint file suffix: control / fine-tune fits never touch the main run's files
        self.tiles = tiles
        self.train_tiles = [t for t in tiles if t.role == "train"] if train_tiles is None else train_tiles
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        t0 = self.train_tiles[0] if self.train_tiles else tiles[0]
        self.model = O.OpField(a, reg, gt.meta, ctx.stencil, t0.A0.clone().to(dev), ctx.Phi_nodes, ctx.w_nodes, ctx.lam_nodes, len(ctx.dets), ctx.scale_ch, dev, det_names=ctx.dets)
        m = self.model
        self.train_eids = sorted({e for t in self.train_tiles for e in t.train_eids})
        self.moves = m.set_train_rows(self.train_eids)
        if self.moves:
            log(f"[{tag}] base rows with no training exposure (--base-per {m.base_per}) aliased to a trained row: " + "; ".join(self.moves))
        if m.two_scale and getattr(m, "untrained_wing_cells", None):
            log(f"[{tag}] wing tables with no training exposure aliased: " + "; ".join(m.untrained_wing_cells))
        self.K_x = int(a.xfit_reg)
        self.skies, self.cur, self.pool, self.sampler = {}, None, None, None
        npix = np.zeros(gt.n)
        for t in self.train_tiles:
            for e, info in t.exp_info.items():
                if e in set(t.train_eids):
                    npix[e] += info["npix"]
        self.tr_all = torch.tensor(self.train_eids, device=dev)
        self.npix_all = torch.tensor(npix[self.train_eids], dtype=torch.float32, device=dev)
        ref_inst = a.ref_instrument if a.ref_instrument in reg.instruments else max(reg.instruments, key=lambda s: sum(1 for i in self.train_eids if reg.instrument[int(m.m_band[i])] == s))
        if ref_inst != a.ref_instrument:
            log(f"[{tag}] WARNING: reference instrument {a.ref_instrument} absent; flux gauge uses {ref_inst}")
        self.ref_mask_all = torch.tensor([reg.instrument[int(m.m_band[i])] == ref_inst for i in self.train_eids], device=dev)
        G = int(aj.gauge_exps) if aj.gauge_exps > 0 else 256
        self.gauge_n = G
        with torch.no_grad():
            m2 = []
            for s0 in range(0, len(self.train_eids), 64):
                sl = slice(s0, s0 + 64)
                m2.append(m.gauge_terms(self.tr_all[sl], self.npix_all[sl], self.ref_mask_all[sl])[2])
            self.m2_ref = float(torch.cat(m2).mean().clamp(min=1e-6))
        self.unit_sig = float(m.m_unit[self.tr_all].mean())
        unit = float(np.mean([float(m.m_unit[i]) for i in self.train_eids]))
        self.lr_bkg = a.lr_bkg if a.lr_bkg > 0 else 0.1 * unit
        self.ggen = torch.Generator().manual_seed(a.seed + 17)
        iw = parse_kv(aj.inst_weights, "--inst-weights")
        for k in iw:
            if k not in reg.instruments:
                raise SystemExit(f"--inst-weights: unknown instrument {k!r} (have {reg.instruments})")
        self.lw = None
        if any(abs(v - 1.0) > 1e-12 for v in iw.values()):
            self.lw = torch.tensor([iw.get(reg.instrument[int(m.m_band[i])], 1.0) for i in range(gt.n)], dtype=torch.float32, device=dev)
        # operator optimiser (opfield.train groups without the sky)
        ops = [p for n_, p in m.named_parameters() if not n_.startswith(("sky", "nuis", "pd", "base", "bank", "mu_skip", "Ze", "Zv", "wing"))]
        groups = [{"params": ops, "lr": a.lr_op}, {"params": m.kernel_params(), "lr": a.lr_kernel}, {"params": [m.nuis.z] + ([m.Ze, m.Zv] if m.zh else []), "lr": a.lr_z}]
        if m.two_scale:
            groups.append({"params": [p_ for p_ in (m.wing_tab, m.wing_logit, m.wing_mod) if p_ is not None], "lr": a.lr_kernel * a.wing_lr_mult})
        if m.use_skip:
            groups.append({"params": list(m.mu_skip.parameters()), "lr": a.lr_mu_skip})
        groups.append({"params": [m.nuis.pe, m.pd] + list(m.nuis.stripes.parameters()), "lr": self.lr_bkg})
        self.opt = torch.optim.Adam(groups)
        self.base_lr = [g["lr"] for g in self.opt.param_groups]
        if self.K_x >= 2:    # folds over ALL training exposures (global): opfield.train's rule
            keys = {}
            for i in self.train_eids:
                keys.setdefault((int(m.m_chan[i]), round(ctx.gt.rep[i]["jac"]["theta_deg"])), []).append(i)
            per_ch = {}
            for k_, ids_ in sorted(keys.items()):
                per_ch.setdefault(k_[0], []).append(ids_)
            for ch_, groups_ in per_ch.items():
                if len(groups_) >= self.K_x:
                    for n_, ids_ in enumerate(groups_):
                        for i in ids_:
                            m.m_fold[i] = (n_ + ch_) % self.K_x
                else:
                    for n_, i in enumerate([i for ids_ in groups_ for i in ids_]):
                        m.m_fold[i] = (n_ + ch_) % self.K_x
            log(f"[{tag}] xfit folds: " + str({f: sum(int((m.m_fold[i] == f)) for i in self.train_eids) for f in range(self.K_x)}) + " training exposures per fold")
        for t in self.train_tiles:
            self.sky_for(t)
        n_op = sum(p.numel() for n_, p in m.named_parameters() if not n_.startswith(("sky", "nuis")))
        log(f"[{tag}] parameters: operator {n_op / 1e3:.1f} k (global), nuisance {sum(p.numel() for p in m.nuis.parameters()) / 1e6:.2f} M ({gt.n} exposures), "
            f"sky per tile {self.skies[self.train_tiles[0].key].sky.numel() / 1e6 if self.train_tiles else 0:.2f} M x {len(self.train_tiles)} tiles (on the CPU unless active)")
        m.model_name = tag

    # ---- sky states and swapping
    def lr_sky_of(self, t):
        return self.a.lr_sky if self.a.lr_sky > 0 else 0.3 * t.sig_mean

    def sky_for(self, t, fresh=False):
        if fresh or t.key not in self.skies:
            self.skies[t.key] = SkyState(t, self.a, self.K_x if (t.role == "train" and not fresh) else 0, self.lr_sky_of(t))
        return self.skies[t.key]

    def activate(self, t, pool=True, chunk=None):
        """Make `t` the tile of the model: its sky + Adam on the device, its b0, its wing FFT size, its sig_b / masks; its pool (a chunk) and PeakSampler when pool=True.
        Everything of the previously active tile leaves the GPU."""
        if self.cur is not None and self.cur is not t:
            self.deactivate()
        m, dev, a = self.model, self.dev, self.a
        if self.cur is not t:
            st = self.sky_for(t).to(dev)
            m.sky = st.sky
            m.m_b0.copy_(t.b0_vec.to(dev))
            if m.two_scale:
                H, W = t.A0.shape[-2:]
                m.wing_Hp, m.wing_Wp = O.next_fast_len(H + 2 * m.wing_nh), O.next_fast_len(W + 2 * m.wing_nh)
            self.sig_b, self.src_masks, self.sig_mean = t.sig_b.to(dev), t.src_masks.to(dev), t.sig_mean
            self.cur = t
        if pool:
            self.load_pool(t, chunk)

    def load_pool(self, t, chunk=None):
        dev = self.dev
        store = t.get_store()
        c = (t.visits if chunk is None else chunk) % t.chunks
        idx = None if t.chunks == 1 else store["perm"][c::t.chunks].long()
        self.free_pool()
        self.pool = GPool(store, idx, dev)
        self.sampler = None
        if getattr(self.a, "importance_sampling", None):
            self.sampler = O.PeakSampler(self.pool, self.model, O.parse_is_radii(self.a.importance_sampling), self.a, self.sig_b)
            self.sampler.refresh(0)
            self.vis_steps = 0
        t.visits += 1
        t.last_pool_bytes = self.pool.nbytes()

    def free_pool(self):
        if self.sampler is not None:
            t = self.cur
            if t is not None:
                ess = [x[1] for x in self.sampler.ess_hist]
                prev = getattr(t, "is_stats", {"ess": [], "refreshes": 0})
                prev["ess"] += ess
                prev["refreshes"] += self.sampler.n_refresh
                t.is_stats = prev
        self.pool, self.sampler = None, None

    def deactivate(self):
        t = self.cur
        if t is None:
            return
        self.free_pool()
        st = self.skies[t.key]
        for p in [st.sky] + st.views:
            p.grad = None
        st.to(torch.device("cpu"))
        self.cur = None
        self.sig_b = self.src_masks = None
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- batches and losses
    def draw(self, B, gen):
        if self.sampler is not None:
            if self.vis_steps > 0 and self.vis_steps % max(1, int(self.a.is_refresh)) == 0:
                self.sampler.refresh(self.vis_steps)
            self.vis_steps += 1
            return self.sampler.draw(B, self.a.bright_frac, gen)
        return self.pool.draw(B, self.a.bright_frac, gen)

    def data_loss(self, bt, out_, p, noise_start):
        if p < noise_start:
            nll, _ = O.nll_terms(bt["d"], out_["pred"], bt["sp"], torch.zeros_like(out_["h"]), torch.full_like(out_["pi"], O.PI_INIT))
        else:
            nll, _ = O.nll_terms(bt["d"], out_["pred"], bt["sp"], out_["h"], out_["pi"])
        if self.lw is not None:
            nll = nll * self.lw[bt["eid"]]
        return 2.0 * (nll.mean() if "w" not in bt else (bt["w"] * nll).mean()), nll

    def sky_regs(self):
        a, m = self.a, self.model
        S = m.band_image()
        r = a.lam_tv * V.tv(S, self.sig_b)
        if a.lam_neg > 0:
            r = r + a.lam_neg * F.relu(-S / self.sig_b[:, None, None] - 2.0).pow(2).mean()
        if a.spec_smooth > 0:
            A = m.sky
            r = r + a.spec_smooth * ((A[2:] - 2 * A[1:-1] + A[:-2]) / self.sig_mean).pow(2).mean()
        return r

    def stripes_pen(self, eids):
        s = self.model.nuis.stripes
        return (s.r[eids].pow(2).sum() + 4 * s.c[eids].pow(2).sum()) / self.unit_sig ** 2 * 1e-3

    def gauge_subset(self):
        n = len(self.train_eids)
        if n <= self.gauge_n:
            return self.tr_all, self.npix_all, self.ref_mask_all
        idx = torch.randperm(n, generator=self.ggen)[:self.gauge_n].to(self.dev)
        return self.tr_all[idx], self.npix_all[idx], self.ref_mask_all[idx]

    def tile_tr(self, t):
        if not hasattr(t, "_tr_dev") or t._tr_dev[0] is not self.dev:
            t._tr_dev = (self.dev, torch.tensor(t.train_eids, device=self.dev))
        return t._tr_dev[1]

    # ---- the two kinds of step
    def sky_step(self, st, gen, B, p, sc, noise_start):
        """One Adam step on the active tile's sky; everything else frozen by the caller."""
        m = self.model
        if self.K_x >= 2 and p < self.a.xfit_frac and not st.switched:
            self.xfit_sky_step(st, gen, B, p, sc, noise_start)
            return
        levels = m.make_levels(sc)
        bt = self.draw(B, gen)
        out_ = m(bt, m.nuis, levels)
        data, _ = self.data_loss(bt, out_, p, noise_start)
        loss = data + self.sky_regs()
        st.opt.zero_grad(set_to_none=True)
        loss.backward()
        st.opt.step()

    def xfit_sky_step(self, st, gen, B, p, sc, noise_start):
        """Pass 1 of opfield.xfit_step only (view f on its own fold, gradient to the view)."""
        a, m = self.a, self.model
        K = self.K_x
        views = st.views
        lv_own = [m.make_levels(sc, sky=v) for v in views]
        bt = self.pool.draw(B, a.bright_frac, gen)
        fold = m.m_fold[bt["eid"]]
        n_b = fold.shape[0]
        st.opt.zero_grad(set_to_none=True)
        gviews = []
        for f in range(K):
            idx = torch.nonzero(fold == f).squeeze(1)
            if idx.numel() == 0:
                gviews.append(None)
                continue
            sub = {k_: (v_[idx] if torch.is_tensor(v_) and v_.shape[:1] == (n_b,) else v_) for k_, v_ in bt.items()}
            o1 = m(sub, m.nuis, lv_own[f])
            if p < noise_start:
                n1, _ = O.nll_terms(sub["d"], o1["pred"], sub["sp"], torch.zeros_like(o1["h"]), torch.full_like(o1["pi"], O.PI_INIT))
            else:
                n1, _ = O.nll_terms(sub["d"], o1["pred"], sub["sp"], o1["h"], o1["pi"])
            S = m.band_image(sky=views[f])
            l1 = 2.0 * n1.sum() / n_b + (a.lam_tv * V.tv(S, self.sig_b) + (a.lam_neg * F.relu(-S / self.sig_b[:, None, None] - 2.0).pow(2).mean() if a.lam_neg > 0 else 0.0)) / K
            gviews.append(torch.autograd.grad(l1, [views[f]])[0])
        for v, g in zip(views, gviews):
            v.grad = g
        st.opt.step()

    def inner_steps(self, t, K, gen, B, p, sc, noise_start):
        """K sky-only steps on tile t: all parameters except the tile's sky (and views) are requires_grad=False (J6)."""
        st = self.skies[t.key]
        keep = [st.sky] + st.views
        with only_trainable(self.model, keep):
            for _ in range(K):
                self.sky_step(st, gen, B, p, sc, noise_start)

    def joint_step(self, t, gen, B, p, sc, noise_start):
        """One step of the full loss on a fresh batch: operator, nuisances and the tile's sky. Returns (data loss, nll mean, pi mean, gauge penalty)."""
        a, m = self.a, self.model
        st = self.skies[t.key]
        tr_t = self.tile_tr(t)
        if self.K_x >= 2 and p < a.xfit_frac and not st.switched:
            tr_g, npix_g, ref_g = self.gauge_subset()
            data, nll, out_, gp, mu_bar, a_ref, m2 = O.xfit_step(m, self.pool, a, st.views, self.K_x, sc, p, noise_start, self.pool_B, gen, self.sig_b, self.sig_mean, tr_g, npix_g,
                                                                  ref_g, self.m2_ref, self.unit_sig, Both(self.opt, st.opt))
            return float(data.detach()), float(nll.mean().detach()), float(out_["pi"].mean().detach()), float(gp.detach())
        levels = m.make_levels(sc)
        bt = self.draw(B, gen)
        out_ = m(bt, m.nuis, levels)
        data, nll = self.data_loss(bt, out_, p, noise_start)
        reg_t = self.sky_regs()
        pri = m.prior_terms(tr_t)
        tr_g, npix_g, ref_g = self.gauge_subset()
        mu_bar, a_ref, m2 = m.gauge_terms(tr_g, npix_g, ref_g)
        gp = a.gauge_shift * (mu_bar ** 2).sum() + a.gauge_flux * a_ref ** 2 + a.gauge_blur * m2.mean() / self.m2_ref
        reg_t = reg_t + gp + a.lc_l2 * (torch.tanh(out_["lraw"]) ** 2).mean() + a.z_l2 * pri["z"] + a.g_l2 * pri["g"] + a.bank_l2 * pri["bank"] + a.base_l2 * pri["base"]
        reg_t = reg_t + self.stripes_pen(tr_t)
        if m.two_scale:
            reg_t = reg_t + a.wing_l2 * pri["wing"]
        loss = data + reg_t
        self.opt.zero_grad(set_to_none=True)
        st.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        st.opt.step()
        self.last_mu = (mu_bar.detach(), a_ref.detach(), m2.detach())
        return float(data.detach()), float(nll.mean().detach()), float(out_["pi"].mean().detach()), float(gp.detach())

    def set_lr(self, frac, st):
        f = lr_factor(frac)
        for g, lr in zip(self.opt.param_groups, self.base_lr):
            g["lr"] = lr * f
        for k, (g, lr) in enumerate(zip(st.opt.param_groups, st.base_lr)):
            g["lr"] = lr * f * (0.0 if (k == 0 and frac < self.a.sky_warmup) else 1.0)

    def maybe_switch_views(self, t, p):
        """opfield.train: after --xfit-frac of the budget the views are averaged into the sky and normal training continues (per tile, at its next visit)."""
        st = self.skies[t.key]
        if self.K_x >= 2 and p >= self.a.xfit_frac and not st.switched:
            with torch.no_grad():
                st.sky.copy_(torch.stack([v for v in st.views]).mean(0))
            st.switched = True
            log(f"[{self.tag}] xfit: tile {t.key} views averaged into the sky; normal training continues")

    # ---- snapshots
    def op_state(self):
        sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items() if k != "sky" and k != "m_b0"}
        return sd

    def load_op_state(self, sd):
        res = self.model.load_state_dict(sd, strict=False)
        bad = [k for k in res.missing_keys if k not in ("sky", "m_b0")] + list(res.unexpected_keys)
        if bad:
            raise SystemExit(f"state does not match this model: {bad[:6]}")

    def sky_states_cpu(self):
        return {k: {"sky": st.sky.detach().cpu().clone(), "views": [v.detach().cpu().clone() for v in st.views], "switched": st.switched} for k, st in self.skies.items()}

    def global_params(self):
        return {n: p for n, p in self.model.named_parameters() if not n.startswith(("sky", "nuis", "Ze", "Zv"))}

    def global_snapshot(self):
        d = {n: p.detach().clone() for n, p in self.global_params().items()}
        d.update({"buf:" + n: b.detach().clone() for n, b in self.model.named_buffers() if n != "m_b0"})
        return d

    def global_delta(self, snap):
        cur = self.global_snapshot()
        return max((float((cur[k] - v).abs().max()) for k, v in snap.items() if v.numel()), default=0.0)

    # ---- training
    def train(self, out, LOG, resume=True):
        a, aj, dev, m = self.a, self.aj, self.dev, self.model
        tiles = self.train_tiles
        if not tiles:
            return
        probs = tile_probs(tiles, aj)
        log(f"[{self.tag}] tile sampling probabilities: " + ", ".join(f"{k} {v:.3f}" for k, v in probs.items()))
        LOG.setdefault("tile_probabilities", {})[self.tag] = probs
        steps = a.steps if a.steps > 0 else 10 ** 12
        budget = a.budget_min * 60.0
        K = int(aj.inner_sky)
        B = self.batch_size()
        self.pool_B = B
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        rng = random.Random(a.seed)
        noise_start = a.c2f_frac if a.noise_start < 0 else a.noise_start
        ck = Path(out) / f"ckpt{self.sfx}.pt"
        cyc, el0, order_i, hist, val_hist = 0, 0.0, 0, [], []
        best_score = float("inf")
        if resume and self.tag == "main" and ck.exists():
            st_ = torch.load(ck, map_location=dev, weights_only=False)
            self.load_op_state(st_["op"])
            self.opt.load_state_dict(st_["opt"])
            for k, s in st_["skies"].items():
                t = next(t for t in tiles if t.key == k)
                sk = self.sky_for(t)
                sk.sky.data = s["sky"].clone()
                for v, vv in zip(sk.views, s["views"]):
                    v.data = vv.clone()
                sk.switched = s["switched"]
                sk.opt.load_state_dict(st_["sky_opt"][k])
            for t in tiles:
                t.cycles, t.visits = st_["tile_cycles"][t.key], st_["tile_visits"][t.key]
            cyc, el0, order_i, hist, val_hist, best_score = st_["cycle"], st_["elapsed"], st_["order_i"], st_["hist"], st_["val_hist"], st_["best_score"]
            rng.setstate(st_["rng"])
            gen.set_state(st_["gen"].cpu())
            log(f"[{self.tag}] resumed from {ck}: cycle {cyc}, {el0 / 60:.1f} min done")
        t_start = time.perf_counter() - el0
        t_ck = t_val = time.perf_counter()
        prog = lambda c, el: max(c / steps, el / budget if budget > 0 else 0.0)  # noqa: E731
        val_tiles = [t for t in tiles if t.val_eids]
        val_every = a.val_every_min * 60.0
        bad = 0
        recent = {t.key: [] for t in tiles}
        sec = {"sky": 0.0, "joint": 0.0, "swap": 0.0}
        visit_left, cur = 0, None
        log(f"[{self.tag}] training: budget {a.budget_min} min, max cycles {a.steps or 'inf'}, K={K} inner sky steps, batch {B}, {len(tiles)} tiles, visit {aj.visit_cycles} cycles, "
            f"lr op {a.lr_op} kernel {a.lr_kernel} z {a.lr_z} bkg {self.lr_bkg:.3g} sky 0.3 x sig_b per tile, c2f {a.c2f_start_arcsec}\" -> 0 over the first {a.c2f_frac}")
        order = list(tiles)
        while True:
            el = time.perf_counter() - t_start
            p = prog(cyc, el)
            if p >= 1.0:
                break
            if visit_left == 0:
                t_s = time.perf_counter()
                if aj.tile_order == "roundrobin":
                    cur = order[order_i % len(order)]
                    order_i += 1
                elif aj.tile_order == "random":
                    cur = order[rng.randrange(len(order))]
                else:
                    cur = rng.choices(order, weights=[probs[t.key] for t in order])[0]
                self.activate(cur)
                self.maybe_switch_views(cur, p)
                visit_left = int(aj.visit_cycles)
                sec["swap"] += time.perf_counter() - t_s
            visit_left -= 1
            sc = O.c2f_sigma(p, a)
            st = self.skies[cur.key]
            self.set_lr(p, st)
            t_s = time.perf_counter()
            if K > 0:
                self.inner_steps(cur, K, gen, B, p, sc, noise_start)
            sec["sky"] += time.perf_counter() - t_s
            t_j = time.perf_counter()
            data, nllm, pim, gpv = self.joint_step(cur, gen, B, p, sc, noise_start)
            sec["joint"] += time.perf_counter() - t_j
            cur.cycles += 1
            cyc += 1
            recent[cur.key].append((data, nllm, pim, gpv))
            self.set_lr(prog(cyc, time.perf_counter() - t_start), st)
            now = time.perf_counter()
            if cyc % a.log_every == 0 or cyc == 1:
                msg = " ".join(f"{k}:{np.mean([x[0] for x in v[-20:]]):.3f}" for k, v in recent.items() if v)
                hist.append([cyc, float(np.mean([x[0] for v in recent.values() for x in v[-5:]]))])
                mu_bar, a_ref, m2 = getattr(self, "last_mu", (torch.zeros(2), torch.zeros(()), torch.zeros(1)))
                log(f"[{self.tag}] cycle {cyc} ({(now - t_start) / 60:.1f} min, p {p:.2f}; sky {sec['sky']:.0f} s joint {sec['joint']:.0f} s swap {sec['swap']:.0f} s) 2NLL by tile {msg} "
                    f"pi {pim:.4f} c2f {sc:.2f}px gauge-pen {gpv:.2e} |mu_bar| {float(mu_bar.norm()):.3f}px a_ref {float(a_ref):+.3f} m2 {float(m2.mean()):.3f}px2"
                    + (f" IS ess {self.sampler.last.get('ess', 0):.0f}/{B}" if self.sampler is not None and self.sampler.last else ""))
            if now - t_ck > a.ckpt_every_min * 60:
                self.save_ckpt(ck, cyc, now - t_start, order_i, hist, val_hist, best_score, rng, gen, tiles)
                t_ck = now
            if val_tiles and val_every > 0 and now - t_val > val_every and p > a.c2f_frac:
                score, L = self.validate(val_tiles, a.val_nuis_steps)
                t_val = time.perf_counter()
                val_hist.append([cyc, (t_val - t_start) / 60, score])
                log(f"[{self.tag}] VAL at cycle {cyc} ({(t_val - t_start) / 60:.1f} min): robust all {L['robust_all']:.4f} src {L['robust_src']:.4f} score {score:.4f} ({L['n_exposures']} exposures)")
                if score < best_score:
                    best_score, bad = score, 0
                    torch.save({"op": self.op_state(), "skies": self.sky_states_cpu(), "cycle": cyc, "score": score}, Path(out) / f"opjoint_best{self.sfx}.pt")
                    log(f"[{self.tag}] new best validation checkpoint")
                else:
                    bad += 1
                    if a.patience and bad >= a.patience:
                        log(f"[{self.tag}] no validation improvement in {bad} checks: early stop")
                        break
                self.activate(cur)
                visit_left = min(visit_left, 0)     # re-load the tile pool at the next cycle (the validation swapped tiles)
        self.save_ckpt(ck, cyc, time.perf_counter() - t_start, order_i, hist, val_hist, best_score, rng, gen, tiles)
        if val_tiles:
            score, L = self.validate(val_tiles, a.val_nuis_steps)
            val_hist.append([cyc, (time.perf_counter() - t_start) / 60, score])
            bp = Path(out) / f"opjoint_best{self.sfx}.pt"
            if bp.exists() and best_score < score:
                sb = torch.load(bp, map_location="cpu", weights_only=False)
                self.load_op_state(sb["op"])
                for k, s in sb["skies"].items():
                    sk = self.skies[k]
                    sk.sky.data = s["sky"].to(sk.sky.device)
                    for v, vv in zip(sk.views, s["views"]):
                        v.data = vv.to(v.device)
                log(f"[{self.tag}] restored best validation checkpoint (score {best_score:.4f} vs final {score:.4f})")
            LOG["val_best_score"] = min(best_score, score)
        LOG.setdefault("train", {})[self.tag] = {"cycles": cyc, "minutes": round((time.perf_counter() - t_start) / 60, 2), "seconds": {k: round(v, 1) for k, v in sec.items()},
                                                  "per_tile_cycles": {t.key: t.cycles for t in tiles}, "batch": B, "val_history": val_hist, "loss_history": hist[-300:]}
        self.deactivate()

    def batch_size(self):
        a = self.a
        if a.batch > 0:
            return int(a.batch)
        if self.dev.type == "cuda":
            free = torch.cuda.mem_get_info()[0] / 2 ** 20
            B = 1024
            wing_mb = O.estimate_wing_memory_mb(len(self.model.wing_groups), self.model.wing_Hp, self.model.wing_Wp, self.model.wing_chunk, len(self.reg.bands)) if self.model.two_scale else 0.0
            hw = self.train_tiles[0].A0.shape[-2:]
            while B < 32768 and O.estimate_batch_memory_mb(B * 2, self.model.nk_max, a.spec_q, a.spec_k, hw=hw, H=a.hidden) + wing_mb < 0.5 * free:
                B *= 2
            return B
        return 2048

    def save_ckpt(self, path, cyc, elapsed, order_i, hist, val_hist, best_score, rng, gen, tiles):
        torch.save({"op": self.op_state(), "opt": self.opt.state_dict(), "skies": self.sky_states_cpu(), "sky_opt": {k: self.skies[k].opt.state_dict() for k in self.skies},
                    "tile_cycles": {t.key: t.cycles for t in tiles}, "tile_visits": {t.key: t.visits for t in tiles}, "cycle": cyc, "elapsed": elapsed, "order_i": order_i,
                    "hist": hist, "val_hist": val_hist, "best_score": best_score, "rng": rng.getstate(), "gen": gen.get_state()}, path)

    def validate(self, val_tiles, steps):
        recs = []
        for t in val_tiles:
            r, *_ = self.score_tile(t, "val", steps, tag="VAL ")
            recs += r
        p = TF.pool_records(recs)
        return p["robust_all"] + (p["robust_src"] if p["robust_src"] == p["robust_src"] else 0.0), p | {"robust_all": p["robust_all"], "robust_src": p["robust_src"]}

    # ---- scoring
    def score_tile(self, t, which, steps, tag="", dump_dir=None, keep=None):
        """opfield.evaluate on tile t's held-out ('hold') or validation ('val') exposures with the active model. Returns (records, per_band, per_instrument, phot, nlls, c2, c2s, robust)."""
        eids = t.hold_eids if which == "hold" else t.val_eids
        if not eids:
            return [], {}, {}, {}, {}, None, None, None
        self.activate(t, pool=False)
        gen = torch.Generator(device=self.dev).manual_seed(77 if which == "val" else 1234)
        exps = ScoredExps(t.scored, self.dev)
        c2, c2s, L, phot, per_exp, nlls = O.evaluate(self.model, exps, eids, self.a, self.src_masks, self.cx.band_index, t.apertures_dev(self.dev) if hasattr(t, "apertures_dev") else self._ap(t), self.reg,
                                                      steps, gen, self.lr_bkg, keep_nuis=keep, tag=f"{tag}{t.key} ", dump_dir=dump_dir)
        recs = [dict(r, tile=t.key, field=t.field) for r in per_exp]
        return recs, L["per_band"], L["per_instrument"], phot, nlls, c2, c2s, {"robust": L["robust"], "robust_src": L["robust_src"]}

    def _ap(self, t):
        if t.apertures is None:
            return None
        return tuple(x.to(self.dev) if torch.is_tensor(x) else x for x in t.apertures)

    def score_image(self, t, image, steps):
        """V.heldout_chi2 of a coadd cube on the tile's held-out exposures (the reference for T1: same pixels, same masks)."""
        if not t.hold_eids:
            return []
        dev = self.dev
        sc = ScoredExps(t.scored, dev)
        exps = [dict(sc[i]) for i in t.hold_eids]
        masks = t.src_masks.to(dev)
        V.heldout_chi2(image.to(dev), exps, list(range(len(exps))), dict(self.cx.band_index), len(self.cx.dets), steps=steps, src_masks=masks)
        recs = [dict(r, tile=t.key, field=t.field, instrument=r.get("instrument", self.reg.instrument[self.reg.index[r["band"]]])) for r in V.heldout_chi2.per_exposure]
        V._LCACHE.clear()
        return recs

    # ---- local-only refit (T5, leave-field-out)
    def local_refit(self, t, steps, tag, LOG=None):
        """Operator frozen. Fresh sky (tile coadd) + the tile's exposure codes z, quadratic backgrounds and stripes (+ the hierarchical codes with --z-hier) by Adam on the learned
        noise NLL with opfield's schedule and coarse-to-fine over the first --c2f-frac of the steps. Verifies that no global parameter moved."""
        a, m, dev = self.a, self.model, self.dev
        snap = self.global_snapshot()
        if self.cur is not None:
            self.deactivate()
        self.sky_for(t, fresh=True)
        self.activate(t, pool=True, chunk=0)
        st = self.skies[t.key]
        tr_t = self.tile_tr(t)
        groups = [{"params": [st.sky], "lr": self.lr_sky_of(t)}, {"params": [m.nuis.z] + ([m.Ze, m.Zv] if m.zh else []), "lr": a.lr_z},
                  {"params": [m.nuis.pe] + list(m.nuis.stripes.parameters()), "lr": self.lr_bkg}]
        opt = torch.optim.Adam(groups)
        base = [g["lr"] for g in groups]
        keep = [p for g in groups for p in g["params"]]
        gen = torch.Generator(device=dev).manual_seed(a.seed + 99)
        B = self.batch_size() if a.batch <= 0 else a.batch
        noise_start = a.c2f_frac if a.noise_start < 0 else a.noise_start
        t0 = time.perf_counter()
        loss_hist = []
        with only_trainable(m, keep):
            for it in range(steps):
                p = it / max(steps, 1)
                sc = O.c2f_sigma(p, a)
                f = lr_factor(p)
                for k, (g, lr) in enumerate(zip(opt.param_groups, base)):
                    g["lr"] = lr * f * (0.0 if (k == 0 and p < a.sky_warmup) else 1.0)
                if t.chunks > 1 and it > 0 and it % 500 == 0:
                    self.load_pool(t, chunk=it // 500)
                levels = m.make_levels(sc)
                bt = self.draw(B, gen)
                out_ = m(bt, m.nuis, levels)
                data, _ = self.data_loss(bt, out_, p, noise_start)
                pri = m.prior_terms(tr_t)
                loss = data + self.sky_regs() + a.z_l2 * pri["z"] + self.stripes_pen(tr_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_hist.append(float(data))
                if it % max(steps // 5, 1) == 0 or it == steps - 1:
                    log(f"[{self.tag}] {tag} {t.key} local fit step {it + 1}/{steps} 2NLL {np.mean(loss_hist[-50:]):.4f} ({time.perf_counter() - t0:.0f} s)")
        delta = self.global_delta(snap)
        log(f"[{self.tag}] {tag} {t.key}: local-only fit done; max |change| of any global parameter or buffer = {delta:.3e}")
        self.free_pool()
        return {"steps": steps, "final_2nll": float(np.mean(loss_hist[-50:])), "global_param_max_abs_delta": delta, "seconds": round(time.perf_counter() - t0, 1)}

    def finetune_all(self, left, cycles, LOG, out):
        """Leave-field-out, part 2 (J16): a NEW Fit over ALL tiles (the left-out ones now training), started from this fit's operator, nuisances and skies, `cycles` cycles of the
        normal loop at --lo-ft-lr-mult x every learning rate, no coarse-to-fine restart, noise head on from the start (when the main run uses it). Returns the new Fit."""
        a, aj = self.a, self.aj
        a_ft = copy.copy(a)
        a_ft.steps, a_ft.budget_min, a_ft.xfit_reg = int(cycles), 0.0, 0
        a_ft.c2f_start_arcsec, a_ft.sky_c2f, a_ft.sky_warmup = 0.0, 0.0, 0.0
        a_ft.noise_start = 0.0 if a.noise_start <= 1.0 else a.noise_start
        a_ft.val_every_min = 0.0
        cx = copy.copy(self.cx)
        cx.a = a_ft
        self.deactivate()
        every = [t for t in self.tiles if t.role in ("train", "leftout")]
        f2 = Fit(cx, every, "lo-finetune", train_tiles=every)
        f2.load_op_state(self.op_state())
        f2.model.set_train_rows(f2.train_eids)     # the state's aliasing buffers belong to the first fit; the left-out cells are trained now
        m_ = float(aj.lo_ft_lr_mult)
        f2.base_lr = [x * m_ for x in f2.base_lr]
        for g_, lr_ in zip(f2.opt.param_groups, f2.base_lr):
            g_["lr"] = lr_
        for t in every:
            st2, st1 = f2.sky_for(t), self.skies[t.key]
            st2.sky.data = st1.sky.detach().cpu().clone()
            st2.base_lr = [x * m_ for x in st2.base_lr]
            st2.lr *= m_
        f2.train(out, LOG, resume=False)
        f2.a = a      # scoring with the run's own evaluation settings (nuisance c2f etc.)
        return f2

    def selfcheck(self, t, K=2):
        """Inner sky steps change the sky and nothing else: every global parameter, buffer, nuisance row and the operator's Adam state are bit-identical afterwards."""
        a, m = self.a, self.model
        self.activate(t)
        self.pool_B = self.batch_size() if a.batch <= 0 else a.batch
        snap = self.global_snapshot()
        nu = {n: p.detach().clone() for n, p in m.named_parameters() if n.startswith(("nuis", "Ze", "Zv"))}
        sky0 = self.skies[t.key].sky.detach().clone()
        opt_state = copy.deepcopy(self.opt.state_dict())
        gen = torch.Generator(device=self.dev).manual_seed(5)
        self.inner_steps(t, K, gen, self.pool_B, 0.5, 0.0, 10.0)
        d_glob = self.global_delta(snap)
        d_nuis = max((float((p.detach() - nu[n]).abs().max()) for n, p in m.named_parameters() if n in nu), default=0.0)
        d_sky = float((self.skies[t.key].sky.detach() - sky0).abs().max())
        same_opt = all(torch.equal(x, y) if torch.is_tensor(x) else x == y for gs, gt_ in ((opt_state["state"], self.opt.state_dict()["state"]),) for k in gs for x, y in zip(
            [gs[k][s] for s in sorted(gs[k])], [gt_[k][s] for s in sorted(gt_[k])]))
        with torch.no_grad():
            self.skies[t.key].sky.copy_(sky0)
        ok = d_glob == 0.0 and d_nuis == 0.0 and d_sky > 0.0 and same_opt
        log(f"SELFCHECK [{self.tag}] tile {t.key}: {K} inner sky steps changed the sky by {d_sky:.2e}, the global parameters/buffers by {d_glob:.1e}, the nuisances by {d_nuis:.1e}, operator Adam state unchanged: {same_opt}: {'ok' if ok else 'FAILED'}")
        return {"ok": bool(ok), "d_sky": d_sky, "d_global": d_glob, "d_nuisance": d_nuis, "operator_adam_unchanged": bool(same_opt)}


# ====================================================================================================================== scoring aggregation
def clean(recs):
    return [{k: v for k, v in r.items() if k != "pulls"} for r in recs]


def pool_by(recs, key):
    out = {}
    for k in sorted({r[key] for r in recs}):
        out[k] = TF.pool_records([r for r in recs if r[key] == k])
    return out


def summarize(recs, a, name):
    return TF.summarize(recs, a, name)


def field_scores(recs, a, label):
    """Pooled + bootstrap for the records, per field and per (field, instrument)."""
    res = {}
    for f in sorted({r["field"] for r in recs}):
        rf = [r for r in recs if r["field"] == f]
        ent = {"pooled": summarize(rf, a, f"{label} field {f}"), "per_instrument": {}}
        for ins in sorted({r["instrument"] for r in rf}):
            ri = [r for r in rf if r["instrument"] == ins]
            ent["per_instrument"][ins] = summarize(ri, a, f"{label} {f} {ins}")
        ent["per_band"] = pool_by(rf, "band")
        res[f] = ent
    return res


def dump_dir_for(out, t, name="opfield"):
    if t.kind == "frames":     # compare_operators.py layout: OUT/heldout_pixels/opfield (the opfield-metrics frames field), OUT/<field>/heldout_pixels/opfield for further frames fields
        d = (Path(out) / "heldout_pixels" / name) if t.opfield_metrics else (Path(out) / t.field / "heldout_pixels" / name)
    else:
        d = Path(out) / f"{t.key.replace(':', '_tile')}" / "heldout_pixels" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ====================================================================================================================== main flow
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
                                 epilog="Every other flag is opfield.py's (opfield.py --help): --args-from, --psi-param, --importance-sampling, --two-scale, --loss, --lambda-gauge, --data-dir, "
                                        "--hst-dir, --holdout-from, --n-test, --n-val, --halo-px, --budget-min, --steps, --batch, --dump-heldout-pixels, --quick, ...")
    ap.add_argument("--fields-config", default=None, metavar="FILE", help="JSON fields config (fields_p1j.json): the fields, their tiles / image directories, per-field opfield args, shifts and "
                    "hold-outs. New fields and instruments are entries of this file. Without it, --fields / --plan below describe an LMC plan + SMACS (opfield's data args) the same way")
    ap.add_argument("--fields", default=None, help="legacy: comma list of fields LMC (needs --plan) and/or SMACS (opfield's data args). Default LMC,SMACS with --plan, else SMACS")
    ap.add_argument("--plan", default=None, help="legacy: plan_tiles.py JSON (LMC tiles)")
    ap.add_argument("--tiles", default=None, help="comma list of plan tile ids to use (default: all)")
    ap.add_argument("--heldout-tiles", type=int, default=0, help="T5: the LAST k selected LMC tiles are never used for the operator")
    ap.add_argument("--heldout-tile-ids", default=None, help="T5: explicit comma list of LMC tile ids (overrides --heldout-tiles)")
    ap.add_argument("--tile-use", choices=("all", "covering"), default="all")
    ap.add_argument("--n-test-tile", type=int, default=1, help="T1 held-out dithers per band per LMC tile (V.select_exposures rule; 0 = none)")
    ap.add_argument("--lmc-n-val", type=int, default=0, help="validation exposures per band per LMC tile (SMACS uses opfield's --n-val)")
    ap.add_argument("--lmc-nuis-steps", type=int, default=0, help="held-out nuisance steps on LMC tiles (0 = --nuis-steps)")
    ap.add_argument("--holdout-field", "--leave-out-field", dest="leave_out_field", default=None, metavar="NAME",
                    help="leave-one-field-out: train on the other fields, then refit ONLY sky / codes / nuisances on NAME (global parameters fixed) and score its T1 exposures; "
                         "with --lo-ft-cycles also a short fine-tune of everything (both are reported)")
    ap.add_argument("--lo-ft-cycles", type=int, default=0, help="leave-field-out: cycles of the optional fine-tune of ALL parameters (left-out field included) after the local-only refit (0 = none)")
    ap.add_argument("--lo-ft-lr-mult", type=float, default=0.3, help="leave-field-out fine-tune: multiplier of every learning rate (no coarse-to-fine restart)")
    ap.add_argument("--inner-sky", type=int, default=2, help="K sky-only steps on the tile before each joint step (0 = opfield's single step per cycle)")
    ap.add_argument("--visit-cycles", type=int, default=25, help="cycles per tile visit (one pool upload / sampler rebuild per visit)")
    ap.add_argument("--tile-order", choices=("weighted", "roundrobin", "random"), default="weighted")
    ap.add_argument("--field-floor", type=float, default=0.2, help="minimum total sampling share of a field")
    ap.add_argument("--weight-power", type=float, default=1.0, help="tile weight = training pixels ** this (1 = proportional)")
    ap.add_argument("--tile-weights", default="", help="KEY=mult,... overrides (KEY = tile key such as LMC:3 / SMACS, or a field name)")
    ap.add_argument("--inst-weights", default="", help="INSTRUMENT=mult,... multipliers of the per-pixel loss (default 1)")
    ap.add_argument("--gpu-pool-gb", type=float, default=3.0, help="GPU memory budget for the active tile's pool (50 B/pixel incl. the sampler); larger tiles are used in random chunks")
    ap.add_argument("--pool-chunks", type=int, default=0, help="force the number of pool chunks per tile (0 = from --gpu-pool-gb)")
    ap.add_argument("--tile-store", choices=("ram", "disk"), default="ram")
    ap.add_argument("--gauge-exps", type=int, default=0, help="training exposures per gauge evaluation (0 = 256; all of them when fewer)")
    ap.add_argument("--no-apertures", dest="apertures", action="store_false", help="skip the coadd-peak aperture photometry of evaluate")
    ap.add_argument("--shifts-from", default=None, metavar="FILE", help="A13: apply the per-exposure shifts (mas) of a register_xcorr.py JSON (header CRVAL too)")
    ap.add_argument("--shifts-from-by-tile", action="store_true")
    ap.add_argument("--score-coadds", action="store_true", help="also score the SWarp-like and drizzle coadds on every tile's held-out exposures")
    ap.add_argument("--ht-steps", type=int, default=1500, help="T5: Adam steps of the local-only fit of a held-out tile")
    ap.add_argument("--ht-control", action="store_true", help="T5 control: also refit the OPERATOR on that tile alone from the generic start")
    ap.add_argument("--ht-control-cycles", type=int, default=0, help="cycles of the control (0 = mean cycles of a training tile, >= 300)")
    ap.add_argument("--lo-steps", type=int, default=6000, help="leave-field-out: Adam steps of the local-only refit on the left-out field")
    ap.add_argument("--selfcheck", action="store_true", help="before training: inner sky steps must not move any global parameter (stops on failure)")
    ap.add_argument("--no-resume", action="store_true", help="ignore OUT/ckpt.pt")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    aj, rest = ap.parse_known_args(argv)
    a = O.parse_args(rest)
    if aj.fields_config:
        aj.field_specs = load_fields_config(aj.fields_config)
    else:
        names = [f.strip() for f in (aj.fields.split(",") if aj.fields else (["LMC", "SMACS"] if aj.plan else ["SMACS"])) if f.strip()]
        for f in names:
            if f not in ("LMC", "SMACS"):
                ap.error(f"unknown field {f!r} (LMC, SMACS; any other field needs --fields-config)")
        if "LMC" in names and not aj.plan:
            ap.error("--fields LMC needs --plan")
        aj.field_specs = [({"name": "SMACS", "type": "frames"} if f == "SMACS" else {"name": "LMC", "type": "plan", "plan": aj.plan}) for f in sorted(names, key=lambda x: x != "SMACS")]
    aj.fields_list = [s_["name"] for s_ in aj.field_specs]
    if aj.leave_out_field and aj.leave_out_field not in aj.fields_list:
        ap.error(f"--leave-out-field {aj.leave_out_field} is not one of the fields {aj.fields_list}")
    if aj.leave_out_field and len(aj.fields_list) < 2:
        ap.error("--leave-out-field needs at least two fields")
    if aj.shifts_from and a.coarse_shift_jwst > 0:
        ap.error("--shifts-from and --coarse-shift-jwst > 0 are mutually exclusive (both set the per-exposure shift)")
    if aj.shifts_from_by_tile and not aj.shifts_from:
        ap.error("--shifts-from-by-tile needs --shifts-from")
    if a.operator != "field":
        ap.error("opjoint supports --operator field only (J9)")
    if a.restart_cycles > 0 or a.eval_only:
        ap.error("--restart-cycles and --eval-only are not supported by opjoint (J9)")
    if a.xfit_reg >= 2 and aj.inner_sky > 0 and a.importance_sampling:
        ap.error("--importance-sampling is not implemented together with --xfit-reg (opfield)")
    return aj, a


def main(argv=None):
    aj, a = parse_args(argv)
    O.LOSS_KIND, O.HUBER_DELTA = a.loss, a.huber_delta
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cubes").mkdir(exist_ok=True)
    (out / "kernels").mkdir(exist_ok=True)
    O.LOGF = open(out / "opjoint.log", "a")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_run = time.perf_counter()
    log(f"opjoint start; torch {torch.__version__}; script sha256 {sha16(__file__)[:16]}; opfield sha256 {sha16(O.__file__)[:16]}; device {dev}; fields {aj.fields_list}")
    LOG = {"args": {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)) for k, v in vars(a).items()}, "opjoint_args": {k: v for k, v in vars(aj).items() if k != "field_specs"},
           "approximations": APPROXIMATIONS + ([APPROX_A13] if aj.shifts_from else []), "timings": {}, "metrics": {}, "references": REFERENCES,
           "provenance": {"opjoint_sha256": sha16(__file__), "opfield_sha256": sha16(O.__file__), "tilefit_sha256": sha16(TF.__file__), "argv": sys.argv, "torch": torch.__version__,
                          "python": platform.python_version(), "host": platform.node(), "device": (torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"),
                          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    try:
        import torchsky
        LOG["provenance"]["torchsky"] = torchsky.__version__
    except Exception:  # noqa: BLE001
        pass
    LOG["field_specs"] = aj.field_specs
    tiles = load_fields(a, aj, out)
    if aj.leave_out_field:
        for t in tiles:
            if t.field == aj.leave_out_field:
                t.role = "leftout"
    ctx = assemble(tiles, a, aj, dev, out)
    LOG["split"] = {t.key: {"train": [t.names_by_eid[e] for e in t.train_eids], "holdout": [t.names_by_eid[e] for e in t.hold_eids], "validation": [t.names_by_eid[e] for e in t.val_eids]} for t in tiles}
    LOG["timings"]["assemble"] = round(ctx.seconds_assemble, 1)
    results = run_all(ctx, LOG, out)
    LOG["timings"]["total_min"] = round((time.perf_counter() - t_run) / 60, 2)
    json.dump(LOG, open(out / "metrics.json", "w"), indent=1, default=float)
    log("done")
    O.LOGF.close()
    O.LOGF = None
    return results


def run_all(ctx, LOG, out, fit=None):
    """Fit (or take a given fit), then every score. Returns the tile_metrics dictionary."""
    a, aj, dev = ctx.a, ctx.aj, ctx.dev
    tiles = ctx.tiles
    train_tiles = [t for t in tiles if t.role == "train"]
    if not train_tiles:
        raise SystemExit("no training tile")
    t_fit = time.perf_counter()
    if fit is None:
        fit = Fit(ctx, tiles, "main")
        if aj.selfcheck:
            sc_ = fit.selfcheck(train_tiles[0], max(aj.inner_sky, 1))
            LOG["selfcheck"] = sc_
            if not sc_["ok"]:
                raise SystemExit("SELFCHECK failed")
            fit.deactivate()
        fit.train(out, LOG, resume=not aj.no_resume)
    LOG["timings"]["fit_min"] = round((time.perf_counter() - t_fit) / 60, 2)
    tm = {"tiles": {}, "per_field": {}, "pooled_T1": None}
    steps_of = lambda t: int(t.spec.get("nuis_steps", aj.lmc_nuis_steps if (t.kind == "plan" and aj.lmc_nuis_steps > 0) else a.nuis_steps))  # noqa: E731
    keep = {}
    all_t1 = []
    recs_by_tile = {}
    # ---- T1 on the training tiles
    for t in train_tiles:
        dd = dump_dir_for(out, t) if a.dump_heldout_pixels else None
        t0 = time.perf_counter()
        recs, pb, pi_, phot, nlls, c2, c2s, rob = fit.score_tile(t, "hold", steps_of(t), tag="TEST ", dump_dir=dd, keep=keep)
        ent = {"role": "train", "field": t.field, "n_train_exposures": t.n_train_exp, "n_train_pix": t.n_pix, "cycles": t.cycles, "heldout_dither_images": len(recs), "seconds": round(time.perf_counter() - t0, 1)}
        if recs:
            ent.update(T1=summarize(recs, a, f"T1 {t.key}"), T1_per_band=pb, T1_per_instrument=pi_, T1_photometry=phot, T1_learned_noise=nlls, T1_per_exposure=clean(recs))
            recs_by_tile[t.key] = recs
            all_t1 += recs
            log(f"T1 {t.key}: S/N>5 chi2 {ent['T1']['chi2_src']:.4f}  all {ent['T1']['chi2_all']:.4f}  ({len(recs)} detector images)")
            if t.opfield_metrics:
                M = LOG["metrics"]
                M["heldout_chi2"], M["heldout_chi2_sources"] = {"opfield": c2}, {"opfield": c2s}
                M["heldout_robust"], M["heldout_robust_sources"] = {"opfield": rob["robust"]}, {"opfield": rob["robust_src"]}
                M["heldout_per_band"], M["heldout_per_instrument"] = {"opfield": pb}, {"opfield": pi_}
                M["heldout_photometry"], M["heldout_learned_noise"] = {"opfield": phot}, {"opfield": nlls}
                M["heldout_per_exposure"] = {"opfield": clean(recs)}
                try:
                    M["heldout_bootstrap"] = {"opfield": V.bootstrap_scores(recs, B=a.boot_b, seed=0)}
                    log(V.boot_line("opfield (SMACS in the joint fit)", M["heldout_bootstrap"]["opfield"]))
                except Exception as exc:  # noqa: BLE001
                    log(f"bootstrap failed: {exc!r}")
                LOG["joint_smacs"] = {"T1": ent["T1"], "per_instrument": pi_, "per_band": pb, "references": REFERENCES}
        if aj.score_coadds and recs is not None:
            for nm, img in (("swarp", t.swarp), ("drizzle", t.driz)):
                rc = fit.score_image(t, img, 20 if a.quick else 150)
                if rc:
                    ent[f"T1_{nm}"] = TF.pool_records(rc)
                    ent[f"T1_{nm}_per_exposure"] = clean(rc)
                    recs_by_tile.setdefault(f"{t.key}:{nm}", rc)
        tm["tiles"][t.key] = ent
    if all_t1:
        tm["pooled_T1"] = summarize(all_t1, a, "T1 pooled")
        tm["per_field"] = field_scores(all_t1, a, "T1")
        if aj.score_coadds:
            tm["per_field_coadd"] = {}
            for nm in ("swarp", "drizzle"):
                rc = [r for k, v in recs_by_tile.items() if k.endswith(":" + nm) for r in v]
                if rc:
                    tm["per_field_coadd"][nm] = field_scores(rc, a, f"T1 {nm}")
    # ---- outputs that need the trained model (kernels, mu fields, kernel diagnostics, cubes)
    write_model_outputs(fit, ctx, keep, LOG, out)
    # ---- leave-field-out
    if aj.leave_out_field:
        lo_recs, lo = [], {"field": aj.leave_out_field, "lo_steps": aj.lo_steps, "tiles": {}}
        for t in [x for x in tiles if x.role == "leftout"]:
            info = fit.local_refit(t, aj.lo_steps, "LEAVE-OUT")
            dd = dump_dir_for(out, t) if a.dump_heldout_pixels else None
            recs, pb, pi_, phot, nlls, c2, c2s, rob = fit.score_tile(t, "hold", steps_of(t), tag="LEAVE-OUT ", dump_dir=dd)
            ent = dict(info, n_train_exposures=t.n_train_exp, n_train_pix=t.n_pix, heldout_dither_images=len(recs))
            if recs:
                ent.update(T1=summarize(recs, a, f"leave-out T1 {t.key}"), T1_per_band=pb, T1_per_instrument=pi_, T1_per_exposure=clean(recs))
                lo_recs += recs
                log(f"LEAVE-OUT {t.key}: S/N>5 chi2 {ent['T1']['chi2_src']:.4f}  all {ent['T1']['chi2_all']:.4f}")
                if t.opfield_metrics:
                    M = LOG["metrics"]
                    M["heldout_chi2"], M["heldout_chi2_sources"] = {"opfield_leave_out": c2}, {"opfield_leave_out": c2s}
                    M["heldout_per_band"], M["heldout_per_instrument"] = {"opfield_leave_out": pb}, {"opfield_leave_out": pi_}
            if aj.score_coadds:
                for nm, img in (("swarp", t.swarp), ("drizzle", t.driz)):
                    rc = fit.score_image(t, img, 20 if a.quick else 150)
                    if rc:
                        ent[f"T1_{nm}"] = TF.pool_records(rc)
            lo["tiles"][t.key] = ent
        if lo_recs:
            lo["per_field"] = field_scores(lo_recs, a, "LEAVE-OUT T1")
        if aj.lo_ft_cycles > 0:
            left = [x for x in tiles if x.role == "leftout"]
            f2 = fit.finetune_all(left, aj.lo_ft_cycles, LOG, out)
            ft_recs = []
            for t in left:
                dd = dump_dir_for(out, t, "opfield_lo_finetune") if a.dump_heldout_pixels else None
                recs, pb, pi_, phot, nlls, c2, c2s, rob = f2.score_tile(t, "hold", steps_of(t), tag="LEAVE-OUT-FT ", dump_dir=dd)
                if recs:
                    lo["tiles"][t.key].update(T1_finetune=summarize(recs, a, f"leave-out+finetune T1 {t.key}"), T1_finetune_per_band=pb, T1_finetune_per_instrument=pi_,
                                              T1_finetune_per_exposure=clean(recs), finetune_cycles=aj.lo_ft_cycles)
                    ft_recs += recs
                    log(f"LEAVE-OUT+FINETUNE {t.key}: S/N>5 chi2 {lo['tiles'][t.key]['T1_finetune']['chi2_src']:.4f}  all {lo['tiles'][t.key]['T1_finetune']['chi2_all']:.4f}")
                    if t.opfield_metrics:
                        LOG["metrics"]["heldout_chi2"]["opfield_leave_out_finetune"], LOG["metrics"]["heldout_chi2_sources"]["opfield_leave_out_finetune"] = c2, c2s
            if ft_recs:
                lo["per_field_finetune"] = field_scores(ft_recs, a, "LEAVE-OUT+FT T1")
            f2.deactivate()
            del f2
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        lo["global_param_max_abs_delta"] = max([v["global_param_max_abs_delta"] for v in lo["tiles"].values()] + [0.0])
        lo["note"] = ("operator, detector terms, bank, noise head and all other tiles' parameters were frozen; only sky, exposure codes (and hierarchical codes with --z-hier) and "
                      "nuisances of the left-out field were fitted. Instruments absent from the training fields keep the generic start kernel.")
        tm["leave_out"] = lo
        LOG["leave_out"] = {k: v for k, v in lo.items() if k != "tiles"} | {"tiles": {k: {kk: vv for kk, vv in v.items() if kk != "T1_per_exposure"} for k, v in lo["tiles"].items()}}
    # ---- T5
    ho = [t for t in tiles if t.role == "heldout"]
    if ho:
        t5, ctl_recs, t5_recs = {}, [], []
        for t in ho:
            info = fit.local_refit(t, aj.ht_steps, "T5")
            dd = dump_dir_for(out, t, "opfield_t5") if a.dump_heldout_pixels else None
            recs, pb, pi_, phot, nlls, c2, c2s, rob = fit.score_tile(t, "hold", steps_of(t), tag="T5 ", dump_dir=dd)
            ent = dict(info, n_train_exposures=t.n_train_exp, n_train_pix=t.n_pix, heldout_dither_images=len(recs))
            if recs:
                ent.update(T5=summarize(recs, a, f"T5 {t.key}"), T5_per_band=pb, T5_per_instrument=pi_, T5_per_exposure=clean(recs))
                t5_recs += recs
                log(f"T5 {t.key}: S/N>5 chi2 {ent['T5']['chi2_src']:.4f}  all {ent['T5']['chi2_all']:.4f}")
            t5[t.key] = ent
        fit.deactivate()
        if aj.ht_control:
            mean_cyc = int(np.mean([x.cycles for x in train_tiles])) if train_tiles else 0
            cyc_c = aj.ht_control_cycles if aj.ht_control_cycles > 0 else max(300, mean_cyc)
            for t in ho:
                a_c = copy.copy(a)
                a_c.steps, a_c.budget_min = cyc_c, 0.0
                cx_c = copy.copy(ctx)
                cx_c.a = a_c
                t_ctl = copy.copy(t)
                t_ctl.role = "train"
                t_ctl.visits = 0
                t_ctl.cycles = 0
                fc = Fit(cx_c, [t_ctl], f"control:{t.key}")
                fc.train(out, LOG, resume=False)
                recs, pb, pi_, phot, nlls, c2, c2s, rob = fc.score_tile(t_ctl, "hold", steps_of(t), tag="T5-control ")
                if recs:
                    t5[t.key].update(T5_control=summarize(recs, a, f"T5 control {t.key}"), T5_control_per_band=pb, T5_control_per_exposure=clean(recs), control_cycles=cyc_c)
                    ctl_recs += recs
                    log(f"T5 control {t.key}: S/N>5 chi2 {t5[t.key]['T5_control']['chi2_src']:.4f}  all {t5[t.key]['T5_control']['chi2_all']:.4f}")
                fc.deactivate()
                del fc
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
        if aj.score_coadds:
            for t in ho:
                for nm, img in (("swarp", t.swarp), ("drizzle", t.driz)):
                    rc = fit.score_image(t, img, 20 if a.quick else 150)
                    if rc:
                        t5[t.key][f"T5_{nm}"] = TF.pool_records(rc)
        tm["T5"] = {"tiles": t5}
        if t5_recs:
            tm["T5"]["pooled"] = summarize(t5_recs, a, "T5 pooled")
        if ctl_recs:
            tm["T5"]["pooled_control"] = summarize(ctl_recs, a, "T5 control pooled")
    fit.deactivate()
    LOG["gpu_memory"] = gpu_mem()
    LOG["opjoint"] = {"fields": ctx.aj.fields_list, "dets": ctx.dets, "n_exposures": ctx.gt.n, "tiles": {t.key: {"field": t.field, "role": t.role, "n_pix": t.n_pix, "chunks": t.chunks,
                      "n_train_exposures": t.n_train_exp, "holdout": len(t.hold_eids), "validation": len(t.val_eids), "cycles": t.cycles, "is": getattr(t, "is_stats", None) and
                      {"ess_mean": float(np.mean(t.is_stats["ess"])) if t.is_stats["ess"] else None, "refreshes": t.is_stats["refreshes"]}} for t in tiles},
                      "shifts_applied": ctx.shifts, "base_rows_aliased_untrained": fit.moves, "param_counts": fit.model.param_counts()}
    json.dump(tm, open(Path(out) / "tile_metrics.json", "w"), indent=1, default=float)
    LOG["tile_metrics"] = {"pooled_T1": tm["pooled_T1"], "per_field": tm["per_field"]}
    return tm


def write_model_outputs(fit, ctx, keep, LOG, out):
    """State files, per-tile cubes, and opfield.write_outputs (effective kernels, mu fields, kernel_diagnostics) on the global exposure table."""
    a, m, dev = ctx.a, fit.model, ctx.dev
    out = Path(out)
    torch.save({"format": STATE_FORMAT, "op": fit.op_state(), "args": LOG["args"]}, out / "opjoint_state.pt")
    torch.save({k: v["sky"] for k, v in fit.sky_states_cpu().items()}, out / "opjoint_skies.pt")
    for t in ctx.tiles:
        hdr = fits.Header({k: v for k, v in t.thead.items()})
        name = t.key.replace(":", "")
        fits.writeto(out / "cubes" / f"{name}_drizzle.fits", t.driz.numpy().astype(np.float32), hdr, overwrite=True)
        if t.key in fit.skies:
            fit.activate(t, pool=False)
            with torch.no_grad():
                fits.writeto(out / "cubes" / f"{name}_sky.fits", m.band_image().cpu().numpy().astype(np.float32), hdr, overwrite=True)
    fit.deactivate()
    train_idx = [i for i in fit.train_eids]
    gexps = ctx.gexps
    with torch.no_grad():
        t0 = next((t for t in fit.train_tiles), None)
        if t0 is not None:
            fit.activate(t0, pool=False)
            try:
                O.write_outputs(m, gexps, train_idx, {k: (v[0], None) for k, v in keep.items()}, t0.thead, a, ctx.reg, out, LOG, dev, t0.sig_b.to(dev))
            except Exception as exc:  # noqa: BLE001
                log(f"write_outputs failed ({exc!r}); kernel dumps skipped")
            fit.deactivate()
    LOG["approximations"] = APPROXIMATIONS + ([APPROX_A13] if ctx.aj.shifts_from else [])


if __name__ == "__main__":
    main()
