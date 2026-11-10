#!/usr/bin/env python3
"""tilefit.py: multi-tile training of ONE shared instrument model, every tile with its OWN sky (experiment P1 of the "one sky, many instruments"
programme; operator-aggregate-design-memo.md, "First experiment"). Arm A = the E2-data instrument of smacs_four_v38 (imported as V).

MODEL (per tile k, exposure i of the tile, detector pixel p):   m_i(p) = B_i(p) + g_i [K_(det,band,roll) * s_k]( geometry_i(p) )  (+ stripes_i(p))
  * sky s_k : ONE SpectralSky (grid, 8 RBF spectral coefficients per pixel) PER TILE, initialised from the tile's drizzle coadd (exposures held out of
    the fit excluded), halo --halo-px. Not shared.
  * SHARED across tiles (one set of parameters; "the instrument"):
      - library-PSF tables of V.LibPSF: one learnable unit-sum table per (detector NAME, band), E2-data (--lib-source gauss --lib-learn-table --lib-res off);
        NRCA* and NRCB* have their own tables; the table is rotated onto each tile's sky grid per (band, detector, roll) PSF group (V's detector frame);
      - per-detector static distortion D (cubic Legendre) and per-detector background pattern pd (quartic);
      - per-exposure (detector image) affine M, t, quadratic background pe, gain log g, and 1/f stripes (row/column offsets): ONE parameter set per exposure
        FILE NAME, used by every training tile in which the exposure appears.
  Loss per tile step: V's per-pixel loss (chi2 or --loss huber) of the tile's pixels / their number, + V's penalties (restricted to the tile's exposures
  for the per-exposure terms) + the sky penalties (spectral smoothness, --lam-tv ...).

TRAINING LOOP (VarPro-lite)   one CYCLE = pick a tile (round-robin or --tile-order random); do --inner-sky K Adam steps on THAT TILE'S SKY ONLY (instrument frozen,
  the PSF kernels of the tile's groups computed once and reused: the sky's linear map is fixed); then ONE joint Adam step on the sky (same optimiser) and
  all shared instrument parameters. The instrument therefore receives gradients from every tile over time; each tile's sky is re-solved against the
  current instrument by the inner steps. The inner sky solve is NOT converged (K Adam steps, not an exact solve): this is an approximation of variable
  projection, the envelope theorem holds only approximately. LR schedule: V's warm-up + cosine on max(cycle fraction, time fraction) for every optimiser;
  sky lr = V's (0.3 x mean coadd sigma, per tile), instrument lr = V's (tables --psf-lr or 1e-2, bkg/stripes 1e-3, geometry 2e-3).
  Adam moments of shared per-exposure rows that are absent from the current tile get zero gradients (momentum keeps decaying and moving them for a few
  steps): the usual minibatch-Adam behaviour, not corrected.

MEMORY POLICY  GPU: the shared parameters + Adam state, every tile's sky parameters + Adam state (3-4 x 3.3 M floats per tile at 640^2 px x 8 coefficients ~ 50 MB),
  and ONE chunk (1/--nchunks, default 4, of the training pixels of the current tile, ~56 B per pixel on the GPU) moved to the GPU per step; the FFT
  convolutions of the PSF groups are gradient-checkpointed per chunk of groups (--no-ckpt-groups turns that off) so the activation memory does not grow
  with the number of PSF groups. CPU: all tiles' pixels in RAM in compact form (float32 + int16 indices, ~32 B per pixel; --tile-store disk keeps only the
  current tile in RAM and the rest in OUT/tile_store/*.pt). Estimates are logged per tile; the GPU peak was NOT measured here (CPU-only development).

HELD-OUT DATA (never seen by the instrument)
  T1  per tile: V._auto_holdout rule (V.select_exposures with n_test = --n-test-tile, 2 dithers per band, round-robin over visits, deterministic). An exposure
      held out in ANY tile is excluded from the training of EVERY tile (so no image used for a score is seen anywhere, through its other regions) and is scored
      in every tile it covers. Scores: V.heldout_chi2 (sky, PSF, distortion and detector pattern frozen; the exposure's affine registration refitted, gain and
      quadratic background solved exactly; stripes by the closed form), V.bootstrap_scores (dither-cluster bootstrap), per tile and pooled.
  T5  --heldout-tiles k (or --heldout-tile-ids): tiles never used for the instrument. After training, a NEW sky (from the tile's drizzle coadd) and NEW
      per-exposure nuisance parameters (affine, background, gain, stripes; not those of any training tile) are fitted on the tile's non-held-out exposures with the
      instrument (tables, D, pd) FROZEN (--ht-steps Adam steps), then the tile's held-out dithers are scored as in T1. --ht-control repeats that fit with the
      tables learnable from the generic start and fresh D/pd (what the tile alone gives: the control for T5).
  Validation: --n-val (V's flag) exposures per band per tile kept out of training for checkpoint selection (--val-every-min / --val-every-cycles, --patience).

APPROXIMATIONS (all repeated in metrics.json["approximations"])
  A1 shared per-exposure affine (M, t) is expressed about the tile-LOCAL pivot c (mean of the exposure's loaded pixel positions, as V); for an exposure used by several
     tiles it is one parameter pair, the same physical transform only to first order in M x (pivot difference): ok while |M| x 600 px << 0.1 px.
  A2 per-exposure constant background b0 is the exposure's median in each tile (tile specific constant); pe/gain/stripes are shared.
  A3 the inner sky solve is K Adam steps (not exact); the joint step moves sky and instrument together.
  A4 per-exposure penalties in a tile step are restricted to the exposures of that tile; the gauge terms (mean affine = 0) and the per-detector terms are global.
  A5 each tile has its own TAN header at its centre (north up); the planner's tile boxes were on one common plane (mas differences).
  A6 PSF groups are per tile ((band, detector, roll bin 0.5 deg, parity)); the PSF tables are per (detector, band) with no spatial variation inside the detector (E2: nk = 1,
     window --psf-window-arcsec); (detector, band) pairs seen only in held-out tiles keep the generic start (listed in the log).
  A7 Data pixel sets: DQ == 0 pixels inside the tile box + a 24-pixel pad (V's loader); tiles planned >= 2 x pad apart share no pixels.
  A8 No aperture catalogue / photometry scores (V.APERTURES is None); only chi2-type scores and (with --dump-heldout-pixels) pixel dumps for T7.
  A9 The inner-step kernel cache is exact (kernels are frozen during inner steps); checkpointing only recomputes.
  A10-A11: see APPROXIMATIONS in the code (held-out-tile fit; Adam moments of absent rows).
  A12 (only with --coarse-shift-jwst R > 0) per-exposure COARSE REGISTRATION, a translation only: for each tile, after loading and the global hold-out selection and before any
     coadd / PSF group / sky initialisation, V.coarse_register_all (smacs_four v31, unchanged, run on the CPU) scans, for every (band, dither) exposure group of THAT tile, a
     translation (x, y) of the group's loaded pixel positions over +-R arcsec (1-pixel grid, then 0.25-pixel fine grid, parabolic refinement; a second pass over +-3 px)
     against a splat coadd of the non-held-out exposures of the nearest-wavelength NIRCam band that shares no visit with the group, blurred to the group's FWHM, by a
     clipped (5 sigma) chi2 with a free linear scale + offset per exposure, on <= 40000 random pixels per detector image. One translation is common to all detectors of a
     dither in that tile (so a per-detector or rotation/scale error is NOT measured here: the joint fit's affine M, t still has to find it). An exposure (detector image,
     file name) that belongs to several tiles gets ONE shift: the mean of its per-tile shifts weighted by its loaded pixel count in each tile (groups the engine could not
     register in a tile, e.g. no cross-visit reference band, do not vote). The shift is converted to mas with the tile pixel scale and added, as the same number of mas, to
     the header-WCS pixel positions tx, ty of that exposure in EVERY tile (all tiles are north-up TAN with the same scale and orientation, A5, so a translation in mas is the
     same translation in every tile's pixels). It is NOT put into the shared affine parameter t, which therefore still starts at 0; the fitted t is a residual about the
     shifted position, and tile pivots c (mean of tx, ty) include the shift. The per-exposure spread (largest distance between two tiles' shifts) is logged and written to
     coarse_shifts.json; > 10 mas means the exposure's error is not a pure translation (or the per-tile estimates are noisy) and the combined shift is only an average.
     The reference coadds contain the header-WCS errors of the reference band (cross-band relative registration: a common offset of all exposures is not measured), the
     scan is a likelihood grid search with no catalogue, held-out-tile exposures are registered and combined like the others, and the engine reports chi2(best)/chi2(0)
     (its own normalisation), not an absolute delta chi2. Disabled by default; with R = 0 nothing in this path runs.
  A13 (only with --shifts-from FILE) per-exposure translation read from a JSON written by scripts/register_xcorr.py (catalogue-free image cross-correlation of the exposures
     against each other, one joint robust solve over all tiles; "exposures": {name: {"shift_mas": [dx, dy], "registered": bool, ...}}). It is applied exactly where, and in
     the same way as, the A12 combined shift: after loading and the global hold-out selection, before any coadd / PSF group / sky initialisation, the shift in mas is converted
     with the tile pixel scale and added to the header-WCS pixel positions tx, ty of that exposure in EVERY tile (north-up TAN, same orientation in every tile, A5); the shared
     affine t still starts at 0. Exposures of the tile that are not in the JSON (or not "registered") are left unshifted and counted in the log. With --shifts-from-by-tile the
     per-tile model shift by_tile[tile]["shift_mas_model"] (register_xcorr --similarity) is used for the tiles that have one. Mutually exclusive with --coarse-shift-jwst > 0
     (both write the same quantity). The register_xcorr shifts are relative to its gauge (mean offset of all exposures = 0 by default; one gauge PER connected component, see
     summary.n_components: with more than one component the relative offset between the components, e.g. the F150W+F277W and the F200W+F356W families, is not measured and
     a warning is logged). They are translations only (a rotation / scale of an exposure is not corrected, A1 still has to absorb it), and a common offset of all exposures is not
     measured (the engine's tile frame is the header frame up to that offset). The same shift is also written into e["header"] (CRVAL1 -= dx/(3.6e6 cos dec), CRVAL2 += dy/3.6e6,
     dx west, dy north; once per header dict) because V.classical builds the swarp / drizzle coadds (sky initialisation, --score-coadds) from that header, not from tx, ty; the
     same is done by A12 now. Without the flag nothing runs.
NOT DONE: arm B (--operator opfield): see OperatorBackend below; HST / other instruments (NIRCam only).

--register-only: load the tiles, run the A12 registration for every tile, write coarse_shifts.json (per exposure: tile(s), shift in mas, chi2 ratio of the scan, spread
across tiles; summary percentiles of |shift| per band, programme and visit) and exit, before any model is built (needs --coarse-shift-jwst R > 0).

OUT: coarse_shifts.json (with --coarse-shift-jwst), shifts_applied.json (with --shifts-from: what was applied to which tile), instrument_state.pt (+ instrument_best.pt), tile_metrics.json, metrics.json, explicit_lib_3d_psf_groups.npy (V's detector-frame kernel dump), cubes/tile<k>_*.fits,
tile<k>/heldout_pixels/... with --dump-heldout-pixels, ckpt.pt (resume), tilefit.log.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

V = None  # the engine module (smacs_four_v38 by default), set by load_engine()
STATE_FORMAT = "tilefit-instrument"

# E2-data flags injected when the user did not give them (arm A). Values None = store_true flag.
E2_PRESET = [("--psf-frame", "detector"), ("--psf-model", "lib"), ("--explicit-psf", "lib"), ("--lib-source", "gauss"), ("--lib-learn-table", None),
             ("--lib-res", "off"), ("--psf-window-arcsec", "3.0"), ("--sample-mode", "bicubic"), ("--spectral", "rbf"), ("--init", "drizzle"),
             ("--skip-implicit", None)]


class OperatorBackend:
    """EXTENSION POINT for arm B (opfield). A backend supplies: (1) the shared operator parameters and their optimiser groups, (2) per tile `prepare(sky)` -> what predict
    needs (here: the convolved basis images of the PSF groups), (3) `predict(prep, batch)` -> model pixels, (4) held-out scoring and state export. LibBackend below is the
    only implementation: V.LibPSF through V.SkyModel. opfield.OpField bundles sky and operator in one module with its own pixel pool, hypernetwork codes z_i and
    importance sampling; using it here needs (a) its sky split into per-tile SpectralSky objects with the hypernetwork shared, (b) its Pool / PeakSampler built per tile
    and swapped like the chunks here, (c) opfield's own fit_nuisance / evaluate for T1/T5. Not done."""


# ----------------------------------------------------------------------------- engine and arguments
def load_engine(name):
    global V
    V = importlib.import_module(name)
    return V


def capture_v_parser():
    """The engine builds its argparse parser inside main(): call main() with parse_args replaced by an exception that carries the parser."""
    class _Cap(Exception):
        pass

    orig = argparse.ArgumentParser.parse_args

    def fake(self, args=None, namespace=None):
        raise _Cap(self)

    argparse.ArgumentParser.parse_args = fake
    try:
        V.main()
    except _Cap as c:
        return c.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError("could not capture the engine's argument parser")


def parse_cli(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True, allow_abbrev=False,
                                 epilog="Every other flag is passed to the engine's parser (smacs_four_v38 --help lists them): --halo-px, --psf-window-arcsec, --scale, "
                                        "--spec-k, --svo-cache, --boot-b, --n-val, --patience, --seed, --dump-heldout-pixels, --loss, --quick, ...")
    ap.add_argument("--plan", required=True, help="plan_tiles.py JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="smacs_four_v38", help="engine module in scripts/ (smacs_four_v38 default; v39/v40 give the same arm A)")
    ap.add_argument("--operator", choices=("lib", "opfield"), default="lib", help="lib = arm A (E2-data tables); opfield = arm B (NOT implemented: see OperatorBackend)")
    ap.add_argument("--preset", choices=("e2data", "none"), default="e2data", help="e2data: inject the E2-data engine flags (--psf-frame detector --psf-model lib --explicit-psf lib "
                    "--lib-source gauss --lib-learn-table --lib-res off --psf-window-arcsec 3.0 --sample-mode bicubic --spectral rbf --init drizzle --skip-implicit) unless given")
    ap.add_argument("--tiles", default=None, help="comma list of plan tile ids to use (default: all)")
    ap.add_argument("--heldout-tiles", type=int, default=0, help="the LAST k selected tiles (plan order = greedy order) are never used for the instrument (T5)")
    ap.add_argument("--heldout-tile-ids", default=None, help="explicit comma list of held-out tile ids (overrides --heldout-tiles)")
    ap.add_argument("--tile-use", choices=("all", "covering"), default="all", help="exposures of a tile: every file overlapping it in the plan, or only the covering ones")
    ap.add_argument("--n-test-tile", type=int, default=2, help="T1 held-out dithers per band per tile (V._auto_holdout rule; 0 = none)")
    ap.add_argument("--steps", type=int, default=2000, help="maximum number of cycles (one cycle = inner sky steps + one joint step on one tile)")
    ap.add_argument("--budget-min", type=float, default=0.0, help="stop after this many minutes of training (0 = no time limit)")
    ap.add_argument("--inner-sky", type=int, default=2, help="K sky-only Adam steps on the tile's sky before each joint step")
    ap.add_argument("--visit-cycles", type=int, default=1, help="cycles per tile visit (matters only with --tile-store disk: one load per visit)")
    ap.add_argument("--tile-order", choices=("roundrobin", "random"), default="roundrobin")
    ap.add_argument("--nchunks", type=int, default=4, help="random pixel chunks per tile; one chunk per step is moved to the GPU (V uses 4)")
    ap.add_argument("--tile-store", choices=("ram", "disk"), default="ram")
    ap.add_argument("--no-ckpt-groups", action="store_true", help="no gradient checkpointing of the PSF-group convolutions")
    ap.add_argument("--val-every-min", type=float, default=0.0)
    ap.add_argument("--val-every-cycles", type=int, default=0)
    ap.add_argument("--val-steps", type=int, default=40, help="registration-refit steps of the periodic validation (the final scores use 150, 20 with --quick)")
    ap.add_argument("--ckpt-every-min", type=float, default=30.0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--ht-steps", type=int, default=800, help="Adam steps of the held-out-tile fit (sky + nuisances, instrument frozen)")
    ap.add_argument("--ht-control", action="store_true", help="also fit each held-out tile with its OWN learnable tables from the generic start (control for T5)")
    ap.add_argument("--score-coadds", action="store_true", help="also score the SWarp-like and Drizzle-like coadds of every tile (reference)")
    ap.add_argument("--selfcheck", action="store_true", help="exactness checks of the kernel cache, the checkpointing and the frozen inner steps before training (cheap; recommended on the first run)")
    ap.add_argument("--coarse-shift-jwst", type=float, default=0.0, help="A12: translation search radius (arcsec) of the per-exposure coarse registration (smacs_four v31 "
                    "coarse_register_all, per tile, shifts combined over tiles by pixel-count weights); 0 = off (code path unchanged)")
    ap.add_argument("--register-only", action="store_true", help="run the coarse registration of every tile, write OUT/coarse_shifts.json and exit (needs --coarse-shift-jwst R > 0)")
    ap.add_argument("--shifts-from", default=None, metavar="FILE", help="A13: apply the per-exposure shifts (mas) of a register_xcorr.py JSON (register_xcorr.json) to the header-WCS pixel positions, "
                    "where A12 would apply its combined shifts; mutually exclusive with --coarse-shift-jwst > 0")
    ap.add_argument("--shifts-from-by-tile", action="store_true", help="with --shifts-from: use by_tile[tile]['shift_mas_model'] (register_xcorr --similarity) where the JSON has one")
    ap.add_argument("--log-every", type=int, default=20, help="cycles between log lines")
    a, rest = ap.parse_known_args(argv)
    if a.shifts_from and a.coarse_shift_jwst > 0:
        ap.error("--shifts-from and --coarse-shift-jwst > 0 are mutually exclusive (both set the per-exposure pixel-position shift)")
    if a.shifts_from_by_tile and not a.shifts_from:
        ap.error("--shifts-from-by-tile needs --shifts-from")
    if a.preset == "e2data":
        for flag, val in E2_PRESET:
            if flag not in rest:
                rest += [flag] + ([val] if val is not None else [])
    elif "--skip-implicit" not in rest:
        rest += ["--skip-implicit"]
    return a, rest


def setup_engine(a_t, rest):
    """Parse the engine flags and replicate main()'s global setup of the engine (smacs_four_v38.main lines after parse_args)."""
    vp = capture_v_parser()
    a = vp.parse_args(rest + ["--out", a_t.out])
    for k, v in vars(a_t).items():
        setattr(a, "tf_" + k, v)
    a.coarse_shift_jwst = a_t.coarse_shift_jwst  # the engine routine reads a.coarse_shift_jwst (JWST radius, arcsec) and a.coarse_shift (HST radius: engine default 0; no HST here)
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    if a.psf_window_arcsec > 0:
        n_w = int(round(a.psf_window_arcsec / a.scale))
        a.psf_n = n_w + 1 - n_w % 2
        V.LOG["psf_window"] = {"window_arcsec": a.psf_window_arcsec, "scale": a.scale, "psf_n": a.psf_n, "realised_arcsec": round(a.psf_n * a.scale, 4)}
    V.configure_psf_frame(a)
    V.validate_v38(a)
    if hasattr(V, "validate_v40"):
        V.validate_v40(a)
    bad = []
    if a.spectral != "rbf":
        bad.append("--spectral rbf is required")
    if not V.LIB_ON or not a.lib_learn_table:
        bad.append("--psf-model lib --explicit-psf lib --lib-learn-table (E2-data tables) are required")
    if getattr(a, "lib_chromatic", "none") != "none":
        bad.append("--lib-chromatic is not supported")
    if a.sky_coeffs != "grid":
        bad.append("--sky-coeffs grid only")
    for flag in ("zip", "uq", "rigid", "zip_capacity"):
        if getattr(a, flag, False):
            bad.append(f"--{flag.replace('_', '-')} is not supported")
    if a.optimizer != "adam" or a.freeze_sky_from or a.freeze_instruments_from or a.freeze_instrument_from or a.load_dir or a.hst_dir or a.extra_jwst_dirs:
        bad.append("--optimizer lbfgs, --freeze-*, --load-dir, --hst-dir, --extra-jwst-dirs are not supported (NIRCam cal files come from the plan)")
    if getattr(a, "load_tables_from", None):
        bad.append("--load-tables-from belongs to the SMACS runs (v40), not to tilefit")
    if bad:
        raise SystemExit("tilefit: " + "; ".join(bad))
    V.LAM_LO, V.LAM_HI = a.lam_lo, a.lam_hi
    V.ZIP_PSF_CONVENTION, V.PSF_CONVENTION = a.zip_psf_convention, a.psf_convention
    V.LOG["psf_convention"] = V.PSF_CONVENTION
    V.args_global = a
    V.CALIBRATE_NOISE = a.calibrate_noise
    V.LOSS_KIND = a.loss
    V.HUBER_DELTA = a.huber_delta if a.loss != "gauss" else None
    V.LAM_NEG = a.lam_neg
    V.LAM_TNV, V.LAM_BAND, V.DIP_LR = a.lam_tnv, a.lam_band, a.dip_lr
    V.TARGET_SIG_PX = a.target_fwhm / 2.3548 / a.scale if a.target_fwhm > 0 else 0.0
    V.LOG["args"] = vars(a)
    out = Path(a.out)
    V.SAVE_DIR = out
    (out / "cubes").mkdir(parents=True, exist_ok=True)
    return a


# ----------------------------------------------------------------------------- tiles
class Tile:
    def __init__(self, rec, role):
        self.rec, self.id, self.role = rec, rec["id"], role
        self.ra, self.dec = rec["ra"], rec["dec"]
        self.files = []
        self.exps = []          # exposure dicts (CPU tensors until trimmed)
        self.psfg = None
        self.sky_state = None
        self.model = None
        self.opt_sky = None
        self.pairs = set()
        self.n_cycles = 0
        self.store = None       # {"chunks": [[batch, ...]] * nchunks, "scored": [exposure dicts]} or a path (disk store)
        self.ntot = 0
        self.meta = {}

    # -- data store (RAM or disk)
    def bundle(self):
        if isinstance(self.store, (str, Path)):
            return torch.load(self.store, map_location="cpu", weights_only=False)
        return self.store

    def put(self, bundle, mode, out):
        if mode == "disk":
            d = Path(out) / "tile_store"
            d.mkdir(exist_ok=True)
            p = d / f"tile{self.id}.pt"
            torch.save(bundle, p)
            self.store = str(p)
        else:
            self.store = bundle


def log(msg):
    V.tick(msg)


def load_tile_exposures(tile, a, dev, links_root):
    """Header-validated loading of the plan's files of one tile with V.load_exposures (extra mode: any NIRCam module, MJD, instrument keys); CPU tensors."""
    size_px = a.tile_size_px
    a_halo = a.halo_px
    thead = V.target_header(tile.ra, tile.dec, size_px + 2 * a_halo, a.scale)
    pad = int(round(24 * 0.04 / a.scale)) - a_halo
    d = Path(links_root) / f"tile{tile.id}"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"):
        f.unlink()
    for f in tile.files:
        os.symlink(f, d / os.path.basename(f))
    ctx = {"glob": "*_cal.fits", "include_miri": False, "skipped": []}
    exps = V.load_exposures(str(d), thead, torch.device("cpu"), pad=pad, extra=ctx)
    for e in exps:
        e["tile"] = tile.id
        e["det"] = a.dets.index(e["detname"])
        e["npix"] = int(e["d"].numel())
    tile.thead = thead
    tile.skipped = ctx["skipped"]
    return exps


def select_holdouts(tiles, a):
    """T1 selection per tile by V.select_exposures, then global exclusion: an exposure held out (or validation) in any tile is excluded from every tile's training."""
    names_h, names_v = set(), set()
    for t in tiles:
        sel = V.select_exposures(t.exps, n_test=max(a.tf_n_test_tile, 1), n_val=a.n_val, no_holdout=a.tf_n_test_tile == 0, bands=V.REG.bands)
        for w in sel["warnings"]:
            log(f"tile {t.id} selection warning: {w}")
        names_h |= set(sel["holdout"])
        names_v |= set(sel["validation"])
    for t in tiles:
        for e in t.exps:
            e["holdout"] = e["name"] in names_h
            e["val"] = (e["name"] in names_v) and not e["holdout"]
    return names_h, names_v


# ----------------------------------------------------------------------------- A12: coarse registration
def _pctl(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "p50": float(np.percentile(v, 50)), "p68": float(np.percentile(v, 68)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def combine_tile_shifts(per_tile):
    """per_tile: list of dicts {tile, shift_mas (x, y), npix, ...} of ONE exposure -> (combined (x, y) mas, spread mas): pixel-count weighted mean, spread = largest distance
    between two tiles' shifts (0 for a single tile)."""
    w = np.array([max(float(r["npix"]), 1.0) for r in per_tile])
    s = np.array([r["shift_mas"] for r in per_tile], float)
    comb = (w[:, None] * s).sum(0) / w.sum()
    spread = max((float(np.hypot(*(s[i] - s[j]))) for i in range(len(s)) for j in range(i + 1, len(s))), default=0.0)
    return comb, spread


def coarse_register_tiles(tiles, a, out=None, apply=True):
    """A12 (module docstring). Runs V.coarse_register_all per tile (CPU), combines the shifts of each exposure over the tiles, applies the combined shift to the tx, ty of
    the exposure in every tile (if `apply`) and returns the coarse_shifts.json dictionary (also written to OUT/coarse_shifts.json when `out` is given)."""
    cpu = torch.device("cpu")
    t0 = time.perf_counter()
    per = {}      # exposure name -> list of per-tile records
    meta = {}
    for t in tiles:
        scale = abs(float(t.thead["CD2_2"])) * 3600.0
        old = {e["name"]: (e["tx"], e["ty"]) for e in t.exps}
        V.coarse_register_all(t.exps, t.thead, cpu, a)
        rec = {k: dict(v) for k, v in V.LOG.get("coarse_shift", {}).items()}
        n_reg = 0
        for e in t.exps:
            r = rec.get(f"{e['band']}|{V.dither_key(e['name'])}")
            e["tx"], e["ty"] = old[e["name"]]   # undo: the combined shift is applied below
            p = V.parse_exposure_name(e["name"]) or {}
            meta[e["name"]] = {"band": e["band"], "detector": e["detname"], "programme": p.get("prog"), "visit": "-".join(map(str, p["visit"])) if p.get("visit") else None,
                               "dither_group": str(V.dither_key(e["name"]))}
            if r is None or e["npix"] == 0:
                continue
            n_reg += 1
            ratios = [r[k] for k in ("chi2_ratio_pass0", "chi2_ratio_pass1") if k in r]
            per.setdefault(e["name"], []).append({"tile": t.id, "shift_mas": [float(r["shift_mas"][0]), float(r["shift_mas"][1])], "npix": int(e["npix"]),
                                                  "chi2_ratio": ratios, "at_grid_edge": bool(r.get("at_grid_edge", False)), "ref_band": r.get("ref"), "scale_arcsec": scale})
        mag = [math.hypot(*r["shift_mas"]) for r in rec.values()]
        log(f"A12 tile {t.id}: {len(rec)} (band, dither) groups registered, {n_reg} of {len(t.exps)} exposures; |shift| median {np.median(mag) if mag else 0:.1f} mas, max {max(mag) if mag else 0:.1f} mas")
    exps_out, spreads = {}, []
    for name, m in meta.items():
        recs = per.get(name, [])
        if recs:
            comb, spread = combine_tile_shifts(recs)
            ratio = float(np.average([r["chi2_ratio"][-1] for r in recs if r["chi2_ratio"]], weights=[r["npix"] for r in recs if r["chi2_ratio"]])) if any(r["chi2_ratio"] for r in recs) else None
        else:
            comb, spread, ratio = np.zeros(2), 0.0, None
        exps_out[name] = dict(m, registered=bool(recs), tiles=[r["tile"] for r in recs], shift_mas=[float(comb[0]), float(comb[1])], abs_shift_mas=float(np.hypot(*comb)),
                              spread_mas=float(spread), chi2_ratio=ratio, by_tile={str(r["tile"]): {"shift_mas": r["shift_mas"], "npix": r["npix"], "chi2_ratio_by_pass": r["chi2_ratio"],
                                                                                                       "at_grid_edge": r["at_grid_edge"], "ref_band": r["ref_band"]} for r in recs})
        if len(recs) > 1:
            spreads.append((spread, name))
    if apply:
        for t in tiles:
            scale = abs(float(t.thead["CD2_2"])) * 3600.0
            for e in t.exps:
                x = exps_out.get(e["name"])
                if x and x["registered"]:
                    e["tx"] = e["tx"] + float(x["shift_mas"][0]) / 1000.0 / scale
                    e["ty"] = e["ty"] + float(x["shift_mas"][1]) / 1000.0 / scale
                    shift_exposure_header(e, t.thead, x["shift_mas"][0], x["shift_mas"][1])   # the coadds (V.classical) read e["header"], not tx, ty
    reg = [x for x in exps_out.values() if x["registered"]]

    def group(key):
        keys = sorted({x[key] for x in reg if x[key] is not None}, key=str)
        return {str(k): _pctl([x["abs_shift_mas"] for x in reg if x[key] == k]) for k in keys}
    summ = {"abs_shift_mas_all": _pctl([x["abs_shift_mas"] for x in reg]), "by_band": group("band"), "by_programme": group("programme"), "by_visit": group("visit"),
            "n_exposures": len(exps_out), "n_registered": len(reg), "n_unregistered": len(exps_out) - len(reg),
            "n_multi_tile": len(spreads), "spread_mas_multi_tile": _pctl([s for s, _ in spreads]), "n_spread_gt_10mas": sum(1 for s, _ in spreads if s > 10.0),
            "worst_spread": [{"exposure": n, "spread_mas": s} for s, n in sorted(spreads, reverse=True)[:10]],
            "n_at_grid_edge": sum(1 for x in reg if any(b["at_grid_edge"] for b in x["by_tile"].values()))}
    res = {"description": "A12 coarse registration: shifts in mas along the tile pixel axes (north-up TAN, same orientation in every tile); the shift to ADD to the header-WCS pixel position "
           "of the exposure. shift_mas = pixel-count-weighted mean over tiles; spread_mas = largest distance between two tiles' shifts; chi2_ratio = chi2(best)/chi2(0) of the engine's "
           "scan (clipped, normalised per detector image), not an absolute delta chi2.", "radius_arcsec": float(a.coarse_shift_jwst), "applied": bool(apply),
           "seconds": round(time.perf_counter() - t0, 1), "summary": summ, "exposures": exps_out}
    log(f"A12 summary: {len(reg)} of {len(exps_out)} exposures registered; |shift| mas p50/p90/p99/max "
        f"{summ['abs_shift_mas_all'].get('p50', 0):.1f}/{summ['abs_shift_mas_all'].get('p90', 0):.1f}/{summ['abs_shift_mas_all'].get('p99', 0):.1f}/{summ['abs_shift_mas_all'].get('max', 0):.1f}; "
        f"{len(spreads)} exposures in >= 2 tiles, spread > 10 mas (not a pure translation): {summ['n_spread_gt_10mas']}")
    for b_, q in summ["by_band"].items():
        log(f"A12 band {b_}: n {q['n']}, |shift| mas p50 {q['p50']:.1f} p90 {q['p90']:.1f} p99 {q['p99']:.1f} max {q['max']:.1f}")
    for w_ in summ["worst_spread"][:5]:
        log(f"A12 spread {w_['spread_mas']:.1f} mas: {w_['exposure']}")
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(Path(out) / "coarse_shifts.json", "w"), indent=1)
    return res


_HDR_SHIFTED = {}   # id(header dict) -> header dict (kept alive so ids are not reused): each header is shifted at most once


def shift_exposure_header(e, thead, dx_mas, dy_mas):
    """Make a WCS built from e["header"] (V.classical -> torchsky coadd_image_plane: the swarp / drizzle coadds, hence the sky initialisation and --score-coadds) agree with the
    shifted tx, ty. The tile-axis shift (dx, dy) [mas] (x to the WEST, y to the north, true = header + shift) is a sky translation of the exposure by dy north and dx west, applied
    to CRVAL1/2 only (SIP / CD / CRPIX untouched): CRVAL1 += -dx / (3.6e6 cos(dec_tile)) deg, CRVAL2 += dy / 3.6e6 deg (CRVAL1 is a rotation about the pole, so the east shift
    dRA cos(dec) is exact at the dec of the tile). Returns True if the header was changed; False if the exposure has no header (synthetic tests, trimmed metadata) or this
    header dict was already shifted (the same dict reached through several exposure dicts / tiles)."""
    hd = e.get("header")
    if not hd or "CRVAL1" not in hd or id(hd) in _HDR_SHIFTED:
        return False
    cosd = math.cos(math.radians(float(thead["CRVAL2"])))
    sc_deg = abs(float(thead["CD2_2"]))
    hd["CRVAL1"] = float(hd["CRVAL1"]) - float(dx_mas) / 3.6e6 / cosd
    hd["CRVAL2"] = float(hd["CRVAL2"]) + float(dy_mas) / 3.6e6
    # refinement: a CRVAL2 change is not an exact translation of the projection (second order, ~0.6 mas for 2.5 arcsec at 45 arcsec from the reference point, dec -69.5): iterate
    # CRVAL1/2 so that the exposure's pivot (mean loaded pixel, else the reference pixel) moves by exactly the shift in the tile frame (astropy chain, same TAN / SIP as torchsky)
    try:
        from astropy.io import fits as _fits
        from astropy.wcs import WCS as _AW
        tw = _AW(_fits.Header({k: v for k, v in thead.items()}))
        if torch.is_tensor(e.get("col")) and e["col"].numel():
            piv = (float(e["col"].double().mean()), float(e["row"].double().mean()))
        else:
            piv = (float(hd["CRPIX1"]) - 1.0, float(hd["CRPIX2"]) - 1.0)
        def chain(crv):
            h2 = dict(hd, CRVAL1=crv[0], CRVAL2=crv[1])
            ra, dec = _AW(_fits.Header(h2), relax=True).all_pix2world(piv[0], piv[1], 0)
            return np.array([float(v) for v in tw.all_world2pix(ra, dec, 0)])
        crv0 = (float(hd["CRVAL1"]) + float(dx_mas) / 3.6e6 / cosd, float(hd["CRVAL2"]) - float(dy_mas) / 3.6e6)    # original reference point
        want = chain(crv0) + np.array([float(dx_mas), float(dy_mas)]) / 1000.0 / (sc_deg * 3600.0)
        crv = [float(hd["CRVAL1"]), float(hd["CRVAL2"])]
        for _ in range(4):
            err = want - chain(crv)                 # tile px still missing
            if np.abs(err).max() < 1e-4:
                break
            crv[0] -= err[0] * sc_deg / cosd        # x increases to the west
            crv[1] += err[1] * sc_deg
        hd["CRVAL1"], hd["CRVAL2"] = crv
    except Exception as exc:  # noqa: BLE001
        log(f"shift_exposure_header: refinement skipped ({type(exc).__name__}: {exc}); first-order CRVAL shift only")
    _HDR_SHIFTED[id(hd)] = hd
    e["hdr_shift_mas"] = [float(dx_mas), float(dy_mas)]
    return True


def apply_external_shifts(tiles, path, by_tile=False, out=None):
    """A13 (module docstring). Reads the per-exposure shifts of a register_xcorr.py JSON and adds them (mas, converted with the tile pixel scale) to tx, ty of the exposures of every
    tile, with the arithmetic of the A12 apply block. Returns a summary dict (also written to OUT/shifts_applied.json when `out` is given)."""
    js = json.load(open(path))
    ex = js["exposures"]
    summ_in = js.get("summary", {})
    n_comp = summ_in.get("n_components", 1)
    if n_comp and n_comp > 1:
        log(f"A13 WARNING: {path} has {n_comp} disconnected components (relative offsets between them are NOT measured, each has its own gauge): "
            f"{[c_.get('bands') for c_ in summ_in.get('components', [])]}")
    applied, unmatched, per_tile, mags = set(), set(), {}, []
    n_hdr = 0
    for t in tiles:
        scale = abs(float(t.thead["CD2_2"])) * 3600.0
        n_t = 0
        for e in t.exps:
            x = ex.get(e["name"])
            if not x or not x.get("registered", True) or x.get("shift_mas") is None:
                unmatched.add(e["name"])
                continue
            sh = x["shift_mas"]
            if by_tile:
                m = (x.get("by_tile") or {}).get(str(t.id), {}).get("shift_mas_model")
                if m is not None:
                    sh = m
            e["tx"] = e["tx"] + float(sh[0]) / 1000.0 / scale
            e["ty"] = e["ty"] + float(sh[1]) / 1000.0 / scale
            n_hdr += int(shift_exposure_header(e, t.thead, sh[0], sh[1]))
            applied.add(e["name"])
            mags.append(math.hypot(float(sh[0]), float(sh[1])))
            n_t += 1
        per_tile[str(t.id)] = {"n_exposures": len(t.exps), "n_shifted": n_t}
        log(f"A13 tile {t.id}: {n_t} of {len(t.exps)} exposures shifted")
    unmatched -= applied
    if not applied:
        raise SystemExit(f"tilefit: --shifts-from {path}: none of the exposures of the loaded tiles is in the file (names are the _cal file basenames)")
    res = {"file": str(path), "by_tile_model": bool(by_tile), "n_applied_exposures": len(applied), "n_unmatched_exposures": len(unmatched), "unmatched_examples": sorted(unmatched)[:10],
           "abs_shift_mas": _pctl(mags), "per_tile": per_tile, "n_headers_shifted": n_hdr, "n_components_in_file": n_comp, "gauge": (js.get("config") or {}).get("gauge")}
    log(f"A13 summary: {len(applied)} exposures shifted, {len(unmatched)} not in the file (unshifted); |shift| mas p50/p90/max "
        f"{res['abs_shift_mas'].get('p50', 0):.1f}/{res['abs_shift_mas'].get('p90', 0):.1f}/{res['abs_shift_mas'].get('max', 0):.1f}")
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(Path(out) / "shifts_applied.json", "w"), indent=1)
    V.LOG["shifts_from"] = res
    return res


def prepare_tile(t, a, dev, lam_nodes, w_nodes, basis, nchunks, gid_of):
    """PSF groups, coadds, sky initialisation, compact chunked pixel store."""
    exps = t.exps
    train_idx = [i for i, e in enumerate(exps) if not e["holdout"] and not e["val"]]
    for b in V.REG.bands:
        if not any(exps[i]["band"] == b for i in train_idx):
            raise SystemExit(f"tile {t.id}: no training exposure in band {b} after the held-out selection (plan more exposures / lower --n-test-tile)")
    V.setup_psf_frame(exps, train_idx, a, dev)
    t.psfg = V.PSFG
    t.n_groups = V.PSFG.n_train
    t0 = time.perf_counter()
    swarp = V.classical(exps, t.thead, "lanczos3", 1.0, dev)
    driz = V.classical(exps, t.thead, "drizzle_square", 0.8, dev) if (a.init == "drizzle" or a.tf_score_coadds) else None  # only computed when used
    stats = [V.band_median_sigma(c, None) for c in swarp]
    sig_b = torch.stack([s_ for _, s_ in stats]).clamp(min=1e-6)
    masks = torch.stack([(c - m_) > 5 * sb for (m_, _), c, sb in zip(stats, swarp, sig_b)])
    if a.halo_px > 0:
        inner = torch.zeros_like(masks[0])
        inner[a.halo_px:a.halo_px + a.tile_size_px, a.halo_px:a.halo_px + a.tile_size_px] = True
        masks = masks & inner
    init = driz if a.init == "drizzle" else swarp
    A0 = V.spectral_init(init, lam_nodes, w_nodes, basis)
    t.sig_b, t.src_masks = sig_b.cpu(), masks.cpu()
    t.swarp, t.driz = swarp.cpu(), (driz.cpu() if driz is not None else None)
    t.A0 = A0.cpu()
    t.sig_mean = float(sig_b.mean())
    # compact chunked store of the training pixels
    g = torch.Generator().manual_seed(1000 + t.id)
    chunks = [[] for _ in range(nchunks)]
    for b in V.REG.bands:
        sel0 = [i for i in train_idx if exps[i]["band"] == b]
        for gp in sorted({exps[i]["pgrp"] for i in sel0}):
            sel = [i for i in sel0 if exps[i]["pgrp"] == gp]
            cat = lambda k: torch.cat([exps[i][k] for i in sel])
            full = {k: cat(k).float() for k in ("tx", "ty", "d", "ivar", "dx", "dy")}
            full["row"], full["col"] = cat("row").short(), cat("col").short()
            full["eidx"] = torch.cat([torch.full((exps[i]["npix"],), gid_of[exps[i]["name"]], dtype=torch.int16) for i in sel])
            full["detv"] = torch.cat([torch.full((exps[i]["npix"],), exps[i]["det"], dtype=torch.int16) for i in sel])
            perm = torch.randperm(full["d"].numel(), generator=g)
            for c in range(nchunks):
                ix = perm[c::nchunks]
                ch = {k: v[ix].contiguous() for k, v in full.items()}
                ch.update(band=b, det=exps[sel[0]]["det"], chunk=c, pgrp=gp)
                chunks[c].append(ch)
            del full
    t.ntot = sum(exps[i]["npix"] for i in train_idx)
    t.train_names = [exps[i]["name"] for i in train_idx]
    t.pairs = {(exps[i]["det"], V.REG.index[exps[i]["band"]]) for i in train_idx}
    # per-exposure constants of the tile: bkg0 and the local affine pivot c
    t.exp_const = {exps[i]["name"]: (exps[i]["bkg0"], float(exps[i]["tx"].mean()), float(exps[i]["ty"].mean())) for i in train_idx}
    scored = [e for e in exps if e["holdout"] or e["val"]]
    for e in scored:
        for k in ("row", "col"):
            e[k] = e[k].long()
    for i in train_idx:  # the training pixels now live in the chunks only
        for k in ("tx", "ty", "d", "ivar", "dx", "dy", "row", "col"):
            exps[i][k] = None
    t.n_scored = len(scored)
    t.scored_names = [e["name"] for e in scored]
    t.holdout_names = [e["name"] for e in scored if e["holdout"]]
    t.val_names = [e["name"] for e in scored if e["val"]]
    t.n_train_exp = len(train_idx)
    t.bundle_obj = {"chunks": chunks, "scored": scored}
    log(f"tile {t.id} [{t.role}]: {len(exps)} exposures loaded ({t.n_train_exp} train, {len(t.holdout_names)} held out, {len(t.val_names)} validation), {t.ntot / 1e6:.2f} Mpx training, "
        f"{t.n_groups} PSF groups, coadds {time.perf_counter() - t0:.0f} s")
    Q = lam_nodes.shape[1]
    p = a.psf_n // 2
    Hp = a.tile_size_px + 2 * a.halo_px + 2 * p
    est_plain = t.n_groups * Q * Hp * Hp * 24 / 2 ** 30
    est_ckpt = (8 * Q * Hp * Hp * 24 + t.n_groups * (a.tile_size_px + 2 * a.halo_px) ** 2 * 4 * 2) / 2 ** 30
    t.meta["gpu_activation_estimate_GiB"] = {"no_checkpoint": round(est_plain, 2), "checkpointed": round(est_ckpt, 2)}
    log(f"tile {t.id}: estimated FFT-convolution activations ~{est_plain:.1f} GiB without and ~{est_ckpt:.1f} GiB with --ckpt-groups (24 B per padded pixel per group-node; NOT measured)")


# ----------------------------------------------------------------------------- the model
def make_tile_model_class():
    class _TileModel(V.SkyModel):
        """V.SkyModel with: penalties restricted to the tile's exposures, PSF-group convolutions with a kernel cache / gradient checkpointing."""

        def __init__(self, sky, psf, bkg, geom, band_index, stripes, sig, ids, ckpt=True):
            super().__init__(sky, psf, bkg, geom, band_index, stripes=stripes, sig=sig)
            self.register_buffer("ids", ids)
            self.ckpt = ckpt

        def penalty(self):
            ids = self.ids
            p = self.psf.penalty()
            g = self.geom
            p = p + g.M[ids].pow(2).sum() * 1e4 + g.t[ids].pow(2).sum() * 1e-2 + g.D.pow(2).sum() * 1e-1
            if g.t.requires_grad:
                p = p + 1e4 * g.t.mean(0).pow(2).sum() + 1e6 * g.M.mean(0).pow(2).sum()
            p = p + 1.0 * self.bkg.logg[ids].pow(2).sum()
            if hasattr(self.sky, "penalty"):
                p = p + self.sky.penalty()
            if self.stripes is not None:
                p = p + (self.stripes.r[ids].pow(2).sum() + 4 * self.stripes.c[ids].pow(2).sum()) / self.sig ** 2 * 1e-3
            return p

        def kernels(self):
            """Kernels of the training PSF groups of the ACTIVE tile (V.PSFG), per chunk of groups (no gradient needed: call under no_grad)."""
            G = V.PSFG.select(None)
            return [V.partial_kernels(V.psf_group_kernels(self.psf, G[a0:a0 + V.PSFG.chunk], self.sky.lam_nodes), V.TARGET_SIG_PX) for a0 in range(0, len(G), V.PSFG.chunk)]

        def prepare_groups(self, A=None, groups=None, Bs=None):
            """V.SkyModel.prepare_groups (spectral grid sky) with optional cached kernels `Bs` and per-chunk gradient checkpointing."""
            if A is not None or groups is not None:
                return super().prepare_groups(A, groups)
            G = V.PSFG.select(None)
            sel = torch.tensor([g["b"] for g in G], dtype=torch.long)
            step = V.PSFG.chunk
            Sn, S = self.sky.nodes()
            q = Sn.shape[1]
            outs = []
            for ci, a0 in enumerate(range(0, len(G), step)):
                Gc, sc = G[a0:a0 + step], sel[a0:a0 + step].to(Sn.device)
                w = self.sky.w_nodes[sc][:, :, None, None, None]

                def fn(Sn_sel, Gc=Gc, w=w, ci=ci):
                    B = Bs[ci] if Bs is not None else V.partial_kernels(V.psf_group_kernels(self.psf, Gc, self.sky.lam_nodes), V.TARGET_SIG_PX)
                    CBn = V.convolve_basis(Sn_sel.flatten(0, 1), B)
                    return (CBn.view(len(Gc), q, *CBn.shape[1:]) * w).sum(1)

                x = Sn[sc]
                outs.append(torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False) if (self.ckpt and torch.is_grad_enabled()) else fn(x))
            return S, torch.cat(outs)

        def prepare(self, A=None, groups=None, Bs=None):
            if Bs is not None or (A is None and groups is None):
                return self.prepare_groups(Bs=Bs)
            return V.SkyModel.prepare(self, A, groups)

    return _TileModel


# ----------------------------------------------------------------------------- shared instrument
class Shared:
    """The instrument shared by every training tile."""

    def __init__(self, a, dev, psf, n_exp, ndet):
        stubs = [{"tx": torch.zeros(1, device=dev), "ty": torch.zeros(1, device=dev), "bkg0": 0.0}] * n_exp
        self.psf = psf
        self.geom = V.Geometry(stubs, ndet, "explicit")
        self.bkg = V.ExplicitBackground(stubs, ndet)
        self.stripes = V.Stripes(n_exp, dev) if a.stripes else None
        self.n_exp, self.ndet = n_exp, ndet
        self.params = list(psf.parameters()) + [self.geom.M, self.geom.t, self.geom.D, self.bkg.pe, self.bkg.pd, self.bkg.logg] + (list(self.stripes.parameters()) if self.stripes else [])
        self.intended = [p.requires_grad for p in self.params]
        self.pair_updates = {}

    def freeze(self):
        for p in self.params:
            p.requires_grad_(False)

    def unfreeze(self):
        for p, r in zip(self.params, self.intended):
            p.requires_grad_(r)

    def groups(self, lrs):
        g = [*V.psf_param_groups(self.psf, lrs["psf"]),
             {"params": [p for p in self.bkg.parameters() if p.requires_grad] + (list(self.stripes.parameters()) if self.stripes is not None else []), "lr": lrs["bkg"]}]
        gp = [p for p in self.geom.parameters() if p.requires_grad]
        if gp:
            g.append({"params": gp, "lr": lrs["geom"]})
        return g

    def tile_views(self, tile, dev, n_exp, gid_of):
        c = torch.zeros(n_exp, 2, device=dev)
        b0 = torch.zeros(n_exp, device=dev)
        for name, (bk, cx, cy) in tile.exp_const.items():
            c[gid_of[name]] = torch.tensor([cx, cy], device=dev)
            b0[gid_of[name]] = bk
        gv = copy.copy(self.geom)
        gv.c = c
        bv = copy.copy(self.bkg)
        bv.b0 = b0
        return gv, bv


def to_dev(ch, dev):
    return {k: ((v.to(dev).long() if k in ("row", "col", "eidx", "detv") else v.to(dev)) if torch.is_tensor(v) else v) for k, v in ch.items()}


def sky_regs(model, S, sig_b, a):
    reg = model.sky.penalty() if hasattr(model.sky, "penalty") else torch.zeros((), device=S.device)
    if a.lam_tv > 0:
        reg = reg + a.lam_tv * V.tv(S, sig_b)
    if V.LAM_TNV > 0:
        reg = reg + V.LAM_TNV * V.tnv(S, sig_b)
    if V.LAM_BAND > 0 and V.TARGET_SIG_PX > 0:
        reg = reg + V.LAM_BAND * V.band_excess(S, sig_b, V.TARGET_SIG_PX)
    if V.LAM_NEG > 0:
        reg = reg + V.LAM_NEG * torch.nn.functional.relu(-S / sig_b[:, None, None] - 2.0).pow(2).mean()
    return reg


def step_core(model, batches, ntot, nchunks, sig_b, a, full_penalty, Bs=None):
    """One backward pass: data loss of `batches` (one chunk) + penalties. Returns the data loss (chi2-like per pixel of the full tile scale)."""
    S, CB = model.prepare(Bs=Bs)
    CBl = CB.detach().requires_grad_(True)
    loss = 0.0
    for e in batches:
        m = model.predict(CBl, None, e)
        li = V.rho(e["ivar"] * (e["d"] - m) ** 2).sum() * nchunks / ntot
        li.backward()
        loss += float(li.detach())
    reg = sky_regs(model, S, sig_b, a)
    if full_penalty:
        reg = reg + model.penalty() - (model.sky.penalty() if hasattr(model.sky, "penalty") else 0.0)
    tens, grads = [CB], [CBl.grad if CBl.grad is not None else torch.zeros_like(CB)]
    if reg.requires_grad:
        tens.append(reg)
        grads.append(torch.ones_like(reg))
    torch.autograd.backward(tens, grads)
    return loss


def inner_sky(cur, bundle, cb, K, nch, a, dev, shared):
    """K Adam steps on the tile's sky only: the shared instrument is frozen (requires_grad off) and the PSF-group kernels are computed once (exact: they do not change)."""
    shared.freeze()
    try:
        with torch.no_grad():
            Bs = cur.model.kernels()
        for k in range(K):
            V._LCACHE.clear()
            batches = [to_dev(b, dev) for b in bundle["chunks"][(cb * (K + 1) + k) % nch]]
            cur.opt_sky.zero_grad(set_to_none=True)
            step_core(cur.model, batches, cur.ntot, nch, cur.sig_dev, a, False, Bs=Bs)
            cur.opt_sky.step()
    finally:
        shared.unfreeze()


def selfcheck(t, shared, a, dev, nch, K):
    """Exactness checks on the first training tile (logged; the run stops on failure): kernel-cached prepare == plain prepare; checkpointed == plain gradients;
    inner sky steps change the sky and NOTHING of the shared instrument."""
    activate(t)
    m = t.model
    with torch.no_grad():
        S0, CB0 = m.prepare()
        Bs = m.kernels()
        S1, CB1 = m.prepare(Bs=Bs)
    e1 = float((CB0 - CB1).abs().max() / CB0.abs().max().clamp(min=1e-20))
    grads = []
    for ck in (True, False):
        m.ckpt = ck
        for p in list(m.sky.parameters()) + [p for p in shared.psf.tab.values()]:
            p.grad = None
        S, CB = m.prepare()
        (CB * torch.linspace(0.5, 1.5, CB.numel(), device=CB.device).view_as(CB)).sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in list(m.sky.parameters()) + [p for p in shared.psf.tab.values()] if p.grad is not None]))
    m.ckpt = not a.tf_no_ckpt_groups
    for p in list(m.sky.parameters()) + [p for p in shared.psf.tab.values()]:
        p.grad = None
    e2 = float((grads[0] - grads[1]).abs().max() / grads[1].abs().max().clamp(min=1e-20))
    snap_i = {id(p): p.detach().clone() for p in shared.params}
    sky0 = [p.detach().clone() for p in m.sky.parameters()]
    sd_opt = copy.deepcopy(t.opt_sky.state_dict())
    inner_sky(t, t.bundle(), 0, K, nch, a, dev, shared)
    d_inst = max(float((p.detach() - snap_i[id(p)]).abs().max()) for p in shared.params)
    d_sky = max(float((p.detach() - q).abs().max()) for p, q in zip(m.sky.parameters(), sky0))
    with torch.no_grad():
        for p, q in zip(m.sky.parameters(), sky0):
            p.copy_(q)
    t.opt_sky.load_state_dict(sd_opt)
    ok = e1 < 1e-5 and e2 < 1e-4 and d_inst == 0.0 and d_sky > 0
    log(f"SELFCHECK tile {t.id}: cached-kernel prepare vs plain max rel diff {e1:.2e}; checkpointed vs plain gradient max rel diff {e2:.2e}; inner sky steps changed the sky by {d_sky:.2e} "
        f"and the shared instrument by {d_inst:.1e}: {'ok' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("SELFCHECK failed")


def lr_factor(frac, steps):
    wu = max(0.03, 20.0 / max(steps, 1))
    return max(frac, 1e-3) / wu if frac < wu else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - wu) / (1 - wu))))


def set_lrs(opts_bases, f):
    for opt, bases in opts_bases:
        for g, lr in zip(opt.param_groups, bases):
            g["lr"] = lr * f


def activate(tile):
    V.PSFG = tile.psfg
    V._LCACHE.clear()


# ----------------------------------------------------------------------------- scoring
def pool_records(recs):
    n = sum(r["n"] for r in recs)
    ns = sum(r["n_src"] for r in recs)
    return {"n_exposures": len(recs), "n_pix": n,
            "chi2_all": sum(r["chi2_sum"] for r in recs) / max(n, 1), "chi2_src": (sum(r["chi2_src_sum"] for r in recs) / ns) if ns else float("nan"),
            "robust_all": sum(r["robust_sum"] for r in recs) / max(n, 1), "robust_src": (sum(r["robust_src_sum"] for r in recs) / ns) if ns else float("nan")}


def score_tile(tile, model, a, dev, which, steps, image=None, dump_name=None):
    """V.heldout_chi2 on the tile's held-out ('hold') or validation ('val') exposures with `model` (or a coadd cube `image`). Returns (records, last, per_band, c2, c2s)."""
    activate(tile)
    bundle = tile.bundle()
    sc = [e for e in bundle["scored"] if (e["holdout"] if which == "hold" else e["val"])]
    if not sc:
        return [], None, None, None, None
    exps = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in e.items()} for e in sc]
    masks = tile.src_masks.to(dev)
    band_index = dict(V.REG.index)
    if dump_name is not None:
        V.set_dump_ctx(a, Path(a.out) / f"tile{tile.id}", dump_name, model)
    try:
        c2, c2s, _ = V.heldout_chi2(image if image is not None else model, exps, list(range(len(exps))), band_index, len(a.dets), steps=steps, src_masks=masks)
    finally:
        if dump_name is not None:
            V.clear_dump_ctx()
    recs = [dict(r, tile=tile.id) for r in V.heldout_chi2.per_exposure]
    last, pb = V.heldout_chi2.last, V.heldout_chi2.last["per_band"]
    V._LCACHE.clear()
    del exps, masks
    return recs, last, pb, c2, c2s


def summarize(recs, a, name):
    if not recs:
        return None
    s = pool_records(recs)
    try:
        bs = V.bootstrap_scores(recs, B=a.boot_b, seed=0)
        s["bootstrap"] = bs
        V.tick(V.boot_line(name, bs))
    except Exception as exc:  # noqa: BLE001
        log(f"bootstrap failed for {name}: {exc!r}")
    return s


# ----------------------------------------------------------------------------- state export
def export_instrument(shared, a, tiles_train, dets, gid_of, path, extra=None):
    psf = shared.psf
    tabs = {}
    for tk, prm in psf.tab.items():
        d, b = (int(x) for x in tk.split("_"))
        th = prm.detach().cpu().clone()
        sp = torch.nn.functional.softplus(th)
        tabs[f"{dets[d]}/{V.REG.bands[b]}"] = {"theta": th, "table": (sp / sp.sum()).clone(), "oversample": int(psf.os), "N": int(th.shape[-1]), "half_window_det_px": float(psf._Rdet.get((d, b), float("nan"))),
                                               "n_updates": int(shared.pair_updates.get((d, b), 0)), "trained": shared.pair_updates.get((d, b), 0) > 0, "requires_grad": bool(prm.requires_grad)}
    D = shared.geom.D.detach().cpu()
    pd = shared.bkg.pd.detach().cpu()
    per_exp = {}
    for name, g in gid_of.items():
        ent = {"M": shared.geom.M[g].detach().cpu().clone(), "t": shared.geom.t[g].detach().cpu().clone(), "pe": shared.bkg.pe[g].detach().cpu().clone(), "logg": float(shared.bkg.logg[g])}
        if shared.stripes is not None:
            ent["stripes_r"], ent["stripes_c"] = shared.stripes.r[g].detach().cpu().clone(), shared.stripes.c[g].detach().cpu().clone()
        per_exp[name] = ent
    st = {"format": STATE_FORMAT, "version": 1, "engine": V.__name__, "dets": list(dets), "bands": list(V.REG.bands), "tables": tabs,
          "geom_D": {dets[i]: D[i].clone() for i in range(len(dets))}, "bkg_pd": {dets[i]: pd[i].clone() for i in range(len(dets))}, "per_exposure": per_exp,
          "psf": {"window_px": int(a.psf_n), "scale_arcsec": a.scale, "lib_source": a.lib_source, "oversample": int(psf.os)},
          "tiles": [{"id": t.id, "ra": t.ra, "dec": t.dec, "n_cycles": t.n_cycles} for t in tiles_train], "args": {k: v for k, v in vars(a).items() if not k.startswith("tf_")}}
    if extra:
        st.update(extra)
    torch.save(st, path)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    a_t, rest = parse_cli(sys.argv[1:] if argv is None else argv)
    load_engine(a_t.engine)
    global TileModelCls
    a = setup_engine(a_t, rest)
    if a_t.operator != "lib":
        raise SystemExit("tilefit: --operator opfield (arm B) is not implemented; see OperatorBackend in the module docstring for what the hook needs")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    V.LOG["device"] = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    plan = json.load(open(a_t.plan))
    recs = plan["tiles"]
    if a_t.tiles:
        keep = {int(x) for x in a_t.tiles.split(",")}
        recs = [r for r in recs if r["id"] in keep]
    if not recs:
        raise SystemExit("no tiles selected")
    if a_t.heldout_tile_ids:
        ho = {int(x) for x in a_t.heldout_tile_ids.split(",")}
    else:
        ho = {r["id"] for r in recs[len(recs) - a_t.heldout_tiles:]} if a_t.heldout_tiles > 0 else set()
    if len(ho) >= len(recs):
        raise SystemExit("every tile is held out")
    tiles = [Tile(r, "heldout" if r["id"] in ho else "train") for r in recs]
    a.tile_size_px = int(round(plan["params"]["size_arcsec"] / a.scale))
    a.size_data, a.size = a.tile_size_px, a.tile_size_px + 2 * a.halo_px
    a.dets = sorted({e["detector"] for r in recs for b in plan["bands"] for e in r["exposures"][b]})
    for t in tiles:
        if a_t.tile_use == "covering":
            t.files = sorted({e["path"] for b in plan["bands"] for e in t.rec["exposures"][b] if e["covering"]})
        else:
            t.files = list(t.rec["files"])
    log(f"plan {a_t.plan}: {len(recs)} tiles ({len(tiles) - len(ho)} train, {len(ho)} held out {sorted(ho)}), tile {plan['params']['size_arcsec']} arcsec = {a.tile_size_px} px at {a.scale} arcsec, "
        f"halo {a.halo_px} px, detectors {a.dets}, device {dev}")
    # ---- phase A: load
    links = out / "tile_links"
    for t in tiles:
        t.exps = load_tile_exposures(t, a, dev, links)
        if t is tiles[0]:
            V.set_registry(V.registry_from_exposures(t.exps, psf_n=a.psf_n))
        miss = [b for b in V.REG.bands if not any(e["band"] == b for e in t.exps)]
        extra_b = sorted({e["band"] for e in t.exps} - set(V.REG.bands))
        if miss or extra_b:
            raise SystemExit(f"tile {t.id}: bands missing {miss} / not in the registry {extra_b}")
        for e in t.exps:
            if e["detname"] not in a.dets:
                raise SystemExit(f"tile {t.id}: detector {e['detname']} not in the plan")
        log(f"tile {t.id}: {len(t.exps)} of {len(t.files)} files loaded ({len(t.skipped)} skipped: {t.skipped[:3]})")
    V.LOG["band_registry"] = V.REG.summary()
    names_h, names_v = select_holdouts(tiles, a)
    log(f"held-out dithers (global, excluded from every training tile): {len(names_h)} detector images; validation {len(names_v)}")
    if a_t.register_only and a.coarse_shift_jwst <= 0:
        raise SystemExit("tilefit: --register-only needs --coarse-shift-jwst R > 0 (arcsec)")
    if a.coarse_shift_jwst > 0:  # A12
        coarse_register_tiles(tiles, a, out, apply=not a_t.register_only)
        if a_t.register_only:
            log(f"--register-only: wrote {out / 'coarse_shifts.json'}; exit")
            return
    if a_t.shifts_from:  # A13
        apply_external_shifts(tiles, a_t.shifts_from, by_tile=a_t.shifts_from_by_tile, out=out)
    # ---- spectral machinery
    lam_nodes, w_nodes, curves = V.load_bandpasses(V.REG.bands, a.spec_q, a.svo_cache)
    lam_nodes, w_nodes = lam_nodes.to(dev), w_nodes.to(dev)
    basis = V.RBFSpectralBasis(a.spec_k, learned=False).to(dev)
    # ---- global exposure ids of the TRAINING tiles
    train_tiles = [t for t in tiles if t.role == "train"]
    gid_of = {}
    for t in train_tiles:
        for e in t.exps:
            if not e["holdout"] and not e["val"]:  # only exposures that train the instrument have shared nuisance parameters
                gid_of.setdefault(e["name"], len(gid_of))
    # ---- phase B: per-tile preparation
    for t in tiles:
        if t.role == "train":
            prepare_tile(t, a, dev, lam_nodes, w_nodes, basis, a_t.nchunks, gid_of)
        else:
            local = {e["name"]: i for i, e in enumerate([e for e in t.exps if not e["holdout"] and not e["val"]])}
            prepare_tile(t, a, dev, lam_nodes, w_nodes, basis, a_t.nchunks, local)
            t.local_gid = local
        t.put(t.bundle_obj, a_t.tile_store, out)
        if a_t.tile_store == "disk":
            t.bundle_obj = None
        else:
            t.bundle_obj = None
        t.exps = [{k: v for k, v in e.items() if not torch.is_tensor(v) and k != "header"} for e in t.exps]  # metadata only
    # ---- shared instrument
    union = V.PSFGroups(a.psf_group_deg, V.PIXEL_BOX)
    union.det_key = V.LIB_ON
    for t in tiles:  # union PSF groups (geometry of every (detector, band) for the library's window scale)
        for g_ in t.psfg.groups:
            k = (g_["b"], g_["det"], round(g_["theta_deg"], 3), g_["parity"])
            if k not in union.index:
                union.index[k] = len(union.groups)
                union.groups.append(dict(g_, id=len(union.groups)))
    pairs_exps = [{"det": d, "band": V.REG.bands[b]} for t in tiles for (d, b) in t.pairs] + [{"det": e["det"], "band": e["band"]} for t in tiles for e in t.exps]
    saved = V.PSFG
    V.PSFG = union
    psf = V.make_libpsf(a, a.dets, pairs_exps, dict(V.REG.index), lam_nodes.cpu(), dev)
    V.PSFG = saved
    psf_start = {k: v.detach().clone() for k, v in psf.state_dict().items()}
    train_pairs = set().union(*[t.pairs for t in train_tiles])
    only_ht = sorted(f"{a.dets[d]}/{V.REG.bands[b]}" for (d, b) in set().union(*[t.pairs for t in tiles]) - train_pairs)
    log(f"shared tables: {len(psf.tab)} (detector, band) pairs, {sum(p.numel() for p in psf.tab.values()) / 1e6:.2f} M table parameters; pairs seen only in held-out tiles (keep the generic start): {only_ht}")
    shared = Shared(a, dev, psf, len(gid_of), len(a.dets))
    Cls = make_tile_model_class()
    band_index = dict(V.REG.index)
    for t in train_tiles:
        gv, bv = shared.tile_views(t, dev, len(gid_of), gid_of)
        coeffs = V.GridSky(t.A0.to(dev), blur_px=a.sky_blur).to(dev)
        sky = V.SpectralSky(coeffs, basis, lam_nodes, w_nodes, lam_smooth=a.spec_smooth, sig_ref=t.sig_mean, relative=a.spec_relative).to(dev)
        ids = torch.tensor([gid_of[n] for n in t.train_names], device=dev)
        t.model = Cls(sky, psf, bv, gv, band_index, shared.stripes, t.sig_mean, ids, ckpt=not a_t.no_ckpt_groups).to(dev)
        t.sig_dev = t.sig_b.to(dev)
        lrs = V.explicit_lrs(a, {"sky": t.sig_mean * 0.3, "psf": 2e-3, "bkg": 1e-3, "geom": 2e-3})
        if a.sky_lr > 0:
            lrs["sky"] = a.sky_lr
        t.opt_sky = torch.optim.Adam(sky.param_groups(lrs["sky"]))
        t.base_sky = [g["lr"] for g in t.opt_sky.param_groups]
        t.lrs = lrs
    lrs0 = train_tiles[0].lrs
    opt_inst = torch.optim.Adam(shared.groups(lrs0))
    base_inst = [g["lr"] for g in opt_inst.param_groups]
    log(f"instrument: {sum(p.numel() for p in shared.params if p.requires_grad) / 1e6:.2f} M trainable parameters; {len(gid_of)} exposures; sky per tile "
        f"{sum(p.numel() for p in train_tiles[0].model.sky.parameters()) / 1e6:.2f} M parameters")
    if a_t.selfcheck:
        selfcheck(train_tiles[0], shared, a, dev, a_t.nchunks, max(a_t.inner_sky, 1))
    # ---- training loop
    rng = random.Random(a.seed if a.seed is not None else 0)
    K, nch = a_t.inner_sky, a_t.nchunks
    budget = a_t.budget_min * 60.0
    steps_total = a_t.steps
    ckpt_path = out / "ckpt.pt"
    c0, el0, order_i = 0, 0.0, 0
    best = None
    bad = 0
    val_hist = []
    if ckpt_path.exists() and not a_t.no_resume:
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        psf.load_state_dict(ck["psf"])
        for n_, m_ in (("geom", shared.geom), ("bkg", shared.bkg)):
            m_.load_state_dict(ck[n_])
        if shared.stripes is not None:
            shared.stripes.load_state_dict(ck["stripes"])
        opt_inst.load_state_dict(ck["opt_inst"])
        for t in train_tiles:
            t.model.sky.load_state_dict(ck["sky"][t.id])
            t.opt_sky.load_state_dict(ck["opt_sky"][t.id])
            t.n_cycles = ck["tile_cycles"][t.id]
        shared.pair_updates = ck["pair_updates"]
        c0, el0, order_i, val_hist = ck["cycle"], ck["elapsed"], ck["order_i"], ck["val_hist"]
        log(f"resumed from {ckpt_path}: cycle {c0}, {el0 / 60:.1f} min done")
    tstart = time.perf_counter() - el0
    t_ck = t_val = time.perf_counter()
    recent = {t.id: [] for t in train_tiles}
    sec_k = {"sky": 0.0, "joint": 0.0, "load": 0.0}

    def snapshot():
        return {"psf": {k: v.detach().cpu().clone() for k, v in psf.state_dict().items()}, "geom": {k: v.detach().cpu().clone() for k, v in shared.geom.state_dict().items()},
                "bkg": {k: v.detach().cpu().clone() for k, v in shared.bkg.state_dict().items()},
                "stripes": {k: v.detach().cpu().clone() for k, v in shared.stripes.state_dict().items()} if shared.stripes is not None else None,
                "sky": {t.id: {k: v.detach().cpu().clone() for k, v in t.model.sky.state_dict().items()} for t in train_tiles}}

    def restore(sn):
        psf.load_state_dict(sn["psf"])
        shared.geom.load_state_dict(sn["geom"])
        shared.bkg.load_state_dict(sn["bkg"])
        if sn["stripes"] is not None:
            shared.stripes.load_state_dict(sn["stripes"])
        for t in train_tiles:
            t.model.sky.load_state_dict(sn["sky"][t.id])

    def validate(steps):
        recs, rob = [], []
        for t in train_tiles:
            which = "val" if any(tt.val_names for tt in train_tiles) else "hold"
            r, *_ = score_tile(t, t.model, a, dev, which, steps)
            recs += r
        p = pool_records(recs) if recs else None
        return recs, p

    log(f"training: up to {steps_total} cycles, budget {a_t.budget_min} min, inner sky steps K={K}, {len(train_tiles)} tiles, chunks {nch}")
    cyc = c0
    visit_left, cur = 0, None
    stop = False
    while cyc < steps_total and not stop:
        if visit_left == 0:
            if a_t.tile_order == "random":
                cur = train_tiles[rng.randrange(len(train_tiles))]
            else:
                cur = train_tiles[order_i % len(train_tiles)]
                order_i += 1
            visit_left = a_t.visit_cycles
            t_l = time.perf_counter()
            bundle = cur.bundle()
            activate(cur)
            sec_k["load"] += time.perf_counter() - t_l
        visit_left -= 1
        activate(cur)
        el = time.perf_counter() - tstart
        prog = max((cyc + 1) / steps_total, el / budget if budget else 0.0)
        f = lr_factor(prog, steps_total)
        set_lrs([(opt_inst, base_inst), (cur.opt_sky, cur.base_sky)], f)
        model = cur.model
        cb = cur.n_cycles
        # ---- inner sky steps
        t_s = time.perf_counter()
        inner_sky(cur, bundle, cb, K, nch, a, dev, shared)
        sec_k["sky"] += time.perf_counter() - t_s
        # ---- joint step
        t_j = time.perf_counter()
        V._LCACHE.clear()
        batches = [to_dev(b, dev) for b in bundle["chunks"][(cb * (K + 1) + K) % nch]]
        opt_inst.zero_grad(set_to_none=True)
        cur.opt_sky.zero_grad(set_to_none=True)
        loss = step_core(model, batches, cur.ntot, nch, cur.sig_dev, a, True)
        opt_inst.step()
        cur.opt_sky.step()
        del batches
        sec_k["joint"] += time.perf_counter() - t_j
        cur.n_cycles += 1
        for pr in cur.pairs:
            shared.pair_updates[pr] = shared.pair_updates.get(pr, 0) + 1
        recent[cur.id].append(loss)
        cyc += 1
        el = time.perf_counter() - tstart
        if cyc % a_t.log_every == 0 or cyc == steps_total or cyc == 1:
            msg = " ".join(f"t{tid}:{np.mean(v[-5:]):.3f}" for tid, v in recent.items() if v)
            log(f"cycle {cyc}/{steps_total} ({el / 60:.1f} min; sky {sec_k['sky']:.0f} s joint {sec_k['joint']:.0f} s load {sec_k['load']:.0f} s) chi2/pix by tile {msg}")
        if budget and el > budget:
            log(f"time budget reached at cycle {cyc}")
            stop = True
        now = time.perf_counter()
        do_val = (a_t.val_every_min > 0 and now - t_val > a_t.val_every_min * 60) or (a_t.val_every_cycles > 0 and cyc % a_t.val_every_cycles == 0)
        if do_val and (a_t.n_test_tile > 0 or a.n_val > 0):
            recs, p = validate(a_t.val_steps)
            t_val = time.perf_counter()
            if p is not None:
                score = p["robust_all"] + (p["robust_src"] if p["robust_src"] == p["robust_src"] else 0.0)
                val_hist.append([cyc, round(el / 3600, 4), p["chi2_all"], p["chi2_src"], p["robust_all"], p["robust_src"]])
                log(f"VAL at cycle {cyc} ({el / 3600:.2f} h): chi2 all {p['chi2_all']:.4f} src {p['chi2_src']:.4f} | robust all {p['robust_all']:.4f} src {p['robust_src']:.4f} ({p['n_exposures']} exposures)")
                if best is None or score < best[0]:
                    best, bad = (score, cyc, snapshot()), 0
                    log(f"new best validation score {score:.4f} at cycle {cyc}")
                else:
                    bad += 1
                    if a.patience and bad >= a.patience:
                        log(f"no validation improvement in {bad} evaluations, stopping at cycle {cyc}")
                        stop = True
        if (now - t_ck > a_t.ckpt_every_min * 60 or stop or cyc >= steps_total):
            torch.save({"psf": psf.state_dict(), "geom": shared.geom.state_dict(), "bkg": shared.bkg.state_dict(), "stripes": shared.stripes.state_dict() if shared.stripes else None,
                        "opt_inst": opt_inst.state_dict(), "sky": {t.id: t.model.sky.state_dict() for t in train_tiles}, "opt_sky": {t.id: t.opt_sky.state_dict() for t in train_tiles},
                        "tile_cycles": {t.id: t.n_cycles for t in train_tiles}, "pair_updates": shared.pair_updates, "cycle": cyc, "elapsed": time.perf_counter() - tstart,
                        "order_i": order_i, "val_hist": val_hist}, ckpt_path)
            t_ck = time.perf_counter()
    V.LOG["tilefit_timings_s"] = {k: round(v, 1) for k, v in sec_k.items()}
    V.LOG["tilefit_cycles_done"] = cyc
    V.LOG["tilefit_val_history"] = val_hist
    if best is not None:
        torch.save(best[2], out / "instrument_best_snapshot.pt")
        restore(best[2])
        log(f"restored the best validation snapshot (score {best[0]:.4f}, cycle {best[1]})")
        V.LOG["tilefit_best"] = {"score": best[0], "cycle": best[1]}
    for p in psf.parameters():
        p.requires_grad_(False)
    # ---- final evaluation
    steps_final = 20 if a.quick else 150
    tm = {"train_tiles": {}, "heldout_tiles": {}}
    all_t1, all_val = [], []
    for t in train_tiles:
        recs, last, pb, c2, c2s = score_tile(t, t.model, a, dev, "hold", steps_final, dump_name="tilefit_lib" if a.dump_heldout_pixels else None)
        ent = {"role": "train", "n_train_exposures": t.n_train_exp, "n_train_pix": t.ntot, "n_groups": t.n_groups, "cycles": t.n_cycles, "heldout_dithers_exposures": len(recs)}
        s = summarize(recs, a, f"T1 tile {t.id}")
        if s:
            ent["T1"] = s
            ent["T1_per_band"] = pb
            ent["T1_per_exposure"] = [{k: v for k, v in r.items() if k != "pulls"} for r in recs]
            all_t1 += recs
        rv, _, pbv, _, _ = score_tile(t, t.model, a, dev, "val", steps_final)
        if rv:
            ent["VAL"] = pool_records(rv)
            all_val += rv
        if a_t.score_coadds:
            for nm, img in (("swarp", t.swarp), ("drizzle", t.driz)):
                if img is None:
                    continue
                rc, *_ = score_tile(t, None, a, dev, "hold", steps_final, image=img.to(dev))
                if rc:
                    ent[f"T1_{nm}"] = pool_records(rc)
        tm["train_tiles"][t.id] = ent
        if s:
            log(f"T1 tile {t.id}: S/N>5 chi2 {s['chi2_src']:.4f}  all {s['chi2_all']:.4f}  ({s['n_exposures']} detector images)")
    tm["pooled_T1"] = summarize(all_t1, a, "T1 pooled")
    if all_val:
        tm["pooled_VAL"] = pool_records(all_val)
    # ---- held-out tiles (T5)
    ho_tiles = [t for t in tiles if t.role == "heldout"]
    all_t5, all_ctl = [], []
    for t in ho_tiles:
        for control in ([False, True] if a_t.ht_control else [False]):
            model, psf_used = ht_fit(t, a, a_t, dev, shared, psf_start, lam_nodes, w_nodes, basis, control)
            nm = "T5 control" if control else "T5"
            recs, last, pb, c2, c2s = score_tile(t, model, a, dev, "hold", steps_final, dump_name=("tilefit_lib_control" if control else "tilefit_lib") if a.dump_heldout_pixels else None)
            s = summarize(recs, a, f"{nm} tile {t.id}")
            ent = tm["heldout_tiles"].setdefault(t.id, {"role": "heldout", "n_train_exposures": t.n_train_exp, "n_train_pix": t.ntot, "n_groups": t.n_groups})
            if s:
                ent["T5_control" if control else "T5"] = s
                ent["T5_control_per_exposure" if control else "T5_per_exposure"] = [{k: v for k, v in r.items() if k != "pulls"} for r in recs]
                ent["T5_control_per_band" if control else "T5_per_band"] = pb
                (all_ctl if control else all_t5).extend(recs)
                log(f"{nm} tile {t.id}: S/N>5 chi2 {s['chi2_src']:.4f}  all {s['chi2_all']:.4f}  ({s['n_exposures']} detector images)")
            if a_t.score_coadds and not control:
                for nm_, img in (("swarp", t.swarp), ("drizzle", t.driz)):
                    if img is None:
                        continue
                    rc, *_ = score_tile(t, None, a, dev, "hold", steps_final, image=img.to(dev))
                    if rc:
                        ent[f"T5_{nm_}"] = pool_records(rc)
            if not control:
                t.ht_sky = model.sky().detach().cpu()
            del model
    if ho_tiles:
        tm["pooled_T5"] = summarize(all_t5, a, "T5 pooled")
        if all_ctl:
            tm["pooled_T5_control"] = summarize(all_ctl, a, "T5 control pooled")
    # ---- outputs
    export_instrument(shared, a, train_tiles, a.dets, gid_of, out / "instrument_state.pt", extra={"pair_updates": {f"{a.dets[d]}/{V.REG.bands[b]}": n for (d, b), n in shared.pair_updates.items()},
                                                                                                    "pairs_only_in_heldout_tiles": only_ht, "heldout_tile_ids": sorted(ho)})
    with torch.no_grad():
        for p in psf.parameters():
            p.requires_grad_(False)
        activate(train_tiles[0])
        np.save(out / "explicit_lib_3d_psf_groups.npy", {"detector_frame": psf.detector_dump()}, allow_pickle=True)
    from astropy.io import fits as _fits
    for t in tiles:
        hdr = _fits.Header(t.thead)
        _fits.writeto(out / "cubes" / f"tile{t.id}_{'drizzle' if t.driz is not None else 'swarp'}.fits", (t.driz if t.driz is not None else t.swarp).numpy().astype(np.float32), hdr, overwrite=True)
        sk = t.model.sky().detach().cpu() if t.model is not None else getattr(t, "ht_sky", None)
        if sk is not None:
            _fits.writeto(out / "cubes" / f"tile{t.id}_sky.fits", sk.numpy().astype(np.float32), hdr, overwrite=True)
    V.LOG["approximations"] = APPROXIMATIONS + ([APPROX_A12] if a.coarse_shift_jwst > 0 else []) + ([APPROX_A13] if a_t.shifts_from else [])
    V.LOG["tilefit"] = {"plan": str(a_t.plan), "heldout_tile_ids": sorted(ho), "tile_centres": {t.id: [t.ra, t.dec] for t in tiles}, "held_out_dithers": sorted(names_h), "validation": sorted(names_v),
                        "n_exposures_shared": len(gid_of), "pairs_only_in_heldout_tiles": only_ht, "gpu_activation_estimates": {t.id: t.meta for t in tiles},
                        "tile_skipped_files": {t.id: t.skipped[:20] for t in tiles}}
    json.dump(tm, open(out / "tile_metrics.json", "w"), indent=1, default=float)
    json.dump(V.LOG, open(out / "metrics.json", "w"), indent=1, default=float)
    log("done")
    return tm


APPROXIMATIONS = [
    "A1 shared per-exposure affine (M, t) about the tile-local pivot: one parameter pair for an exposure used by several tiles (same transform only to first order in M x pivot difference)",
    "A2 per-exposure constant background b0 = the exposure's median in each tile; pe/gain/stripes shared",
    "A3 the inner sky solve is K Adam steps, not exact variable projection; kernels cached (exact) during them",
    "A4 per-exposure penalties restricted to the tile's exposures; gauge terms and per-detector terms global",
    "A5 each tile has its own north-up TAN header at its centre; planner boxes were on one common plane",
    "A6 PSF groups per tile; tables per (detector, band) with no spatial variation in the detector (E2); pairs seen only in held-out tiles keep the generic start",
    "A7 data = DQ==0 pixels in the tile box + 24 px pad; tiles planned >= 2 pads apart share no pixels",
    "A8 no aperture photometry scores (APERTURES None)",
    "A9 global held-out exclusion: an exposure held out in any tile is not used for training in any tile",
    "A10 held-out-tile fit: new sky + new per-exposure nuisances, instrument tables / D / pd frozen; --ht-control: own tables learnable from the generic start",
    "A11 Adam moments of shared per-exposure rows absent from a tile's step decay but keep moving them for a few steps",
]
APPROX_A12 = ("A12 (--coarse-shift-jwst) per-exposure translation from smacs_four coarse_register_all per tile (one shift per (band, dither) group of a tile, likelihood grid scan against the "
              "nearest-wavelength cross-visit coadd, +-R arcsec), combined over tiles by loaded-pixel-count weights per exposure and added (mas) to the header-WCS pixel positions "
              "in every tile; the shared affine t starts at 0 about the shifted position; per-exposure spread across tiles logged (> 10 mas = not a pure translation)")


APPROX_A13 = ("A13 (--shifts-from) per-exposure translation read from a register_xcorr.py JSON (catalogue-free pairwise image cross-correlation, one joint robust least-squares solve "
              "over all tiles; relative to its gauge, one gauge per connected component of the pair graph) and added (mas) to the header-WCS pixel positions of the exposure in every tile "
              "(or the per-tile similarity model with --shifts-from-by-tile), exactly where A12 applies its combined shift; translation only (no rotation / scale), absolute frame not measured, "
              "exposures missing from the file unshifted, mutually exclusive with A12")


def ht_fit(t, a, a_t, dev, shared, psf_start, lam_nodes, w_nodes, basis, control):
    """Held-out tile: see module docstring (T5). Local exposure ids; sky from the tile's coadd."""
    Cls = make_tile_model_class()
    activate(t)
    bundle = t.bundle()
    names = t.train_names
    n = len(names)
    psf = shared.psf
    if control:
        psf = copy.deepcopy(shared.psf)
        psf.load_state_dict(psf_start)
        for p in psf.parameters():
            p.requires_grad_(False)
        for p in psf.tab.values():
            p.requires_grad_(True)
    else:
        for p in psf.parameters():
            p.requires_grad_(False)
    stubs = [{"tx": torch.zeros(1, device=dev), "ty": torch.zeros(1, device=dev), "bkg0": 0.0}] * n
    geom, bkg = V.Geometry(stubs, len(a.dets), "explicit"), V.ExplicitBackground(stubs, len(a.dets))
    stripes = V.Stripes(n, dev) if a.stripes else None
    if not control:
        with torch.no_grad():
            geom.D.copy_(shared.geom.D)
            bkg.pd.copy_(shared.bkg.pd)
        geom.D.requires_grad_(False)
        bkg.pd.requires_grad_(False)
    c = torch.zeros(n, 2, device=dev)
    b0 = torch.zeros(n, device=dev)
    for i, nm in enumerate(names):
        bk, cx, cy = t.exp_const[nm]
        c[i] = torch.tensor([cx, cy], device=dev)
        b0[i] = bk
    geom.c, bkg.b0 = c, b0
    coeffs = V.GridSky(t.A0.to(dev), blur_px=a.sky_blur).to(dev)
    sky = V.SpectralSky(coeffs, basis, lam_nodes, w_nodes, lam_smooth=a.spec_smooth, sig_ref=t.sig_mean, relative=a.spec_relative).to(dev)
    model = Cls(sky, psf, bkg, geom, dict(V.REG.index), stripes, t.sig_mean, torch.arange(n, device=dev), ckpt=not a_t.no_ckpt_groups).to(dev)
    sig_b = t.sig_b.to(dev)
    lrs = V.explicit_lrs(a, {"sky": t.sig_mean * 0.3, "psf": 2e-3, "bkg": 1e-3, "geom": 2e-3})
    if a.sky_lr > 0:
        lrs["sky"] = a.sky_lr
    groups = list(sky.param_groups(lrs["sky"]))
    groups.append({"params": [p for p in bkg.parameters() if p.requires_grad] + (list(stripes.parameters()) if stripes else []), "lr": lrs["bkg"]})
    gp = [p for p in geom.parameters() if p.requires_grad]
    if gp:
        groups.append({"params": gp, "lr": lrs["geom"]})
    if control:
        groups += V.psf_param_groups(psf, lrs["psf"])[1:]
    opt = torch.optim.Adam(groups)
    base = [g["lr"] for g in opt.param_groups]
    steps = a_t.ht_steps if not a.quick else min(a_t.ht_steps, 30)
    nch = a_t.nchunks
    for it in range(steps):
        set_lrs([(opt, base)], lr_factor((it + 1) / steps, steps))
        V._LCACHE.clear()
        batches = [to_dev(b, dev) for b in bundle["chunks"][it % nch]]
        opt.zero_grad(set_to_none=True)
        loss = step_core(model, batches, t.ntot, nch, sig_b, a, True)
        opt.step()
        if it % max(steps // 5, 1) == 0 or it == steps - 1:
            log(f"tile {t.id} held-out-tile fit{' (control)' if control else ''} step {it + 1}/{steps} chi2/pix {loss:.4f}")
    V._LCACHE.clear()
    return model, psf


if __name__ == "__main__":
    main()
