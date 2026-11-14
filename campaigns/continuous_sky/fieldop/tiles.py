"""Several sky patches (tiles, the scoring boxes of a field) under ONE field operator (F32-F36).

  * ONE operator (kernel field, throughput, distortion fields, detector pattern, noise heads, variability scales): global.
  * ONE sky per tile (SkyGrid: coefficients, preconditioner, band noise) with its own Adam; all skies stay on the device (about 90 MB each
    with Adam state at 0.04" for a 20.48" box).
  * ONE row of per-exposure nuisances per exposure (background quadratic, affine, gain u_e, codes z_e and zD_e, stripes), whatever the
    number of tiles it covers. A CUT (data.Exposure) is the part of an exposure inside one tile's data region; cuts of one exposure share
    its global id `eid`.
  * Pixel data of the cuts live on the host (pinned) and are moved to the device per tile; resident tiles are kept in an LRU cache of
    --gpu-resident-gb (a tile is evicted only when a new one would not fit). The data are read-only, so evicting a tile just drops its device
    copies.
  * Tile schedule: a visit = --visit-steps optimisation steps on one tile; tiles are drawn with probability p_k proportional to
    (training pixels of tile k) ^ --tile-weight-power, so the objective is sum_k p_k L_k (L_k the per-pixel mean loss of tile k), not the
    pooled pixel mean unless the power is 1.
  * Gauges with several skies: each connected component of the (exposure, tile) graph has its own common-shift degeneracy (one sky per
    tile can move with every exposure that sees only that component), and each component of the graph of one filter its own pedestal. The
    shift and pedestal projections act per component (one component with one tile: the single-box gauges unchanged).
"""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import torch


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def components(cuts, key=lambda e: None):
    """Connected components of the bipartite (exposure, tile) graph of the given cuts, separately per key(cut) (e.g. the filter).
    Returns {(key, n): sorted list of global eids}."""
    out = {}
    by = {}
    for e in cuts:
        by.setdefault(key(e), []).append(e)
    for k, cs in by.items():
        uf = _UF()
        for e in cs:
            uf.union(("t", e.tile), ("e", e.eid))
        roots = {}
        for e in cs:
            roots.setdefault(uf.find(("e", e.eid)), set()).add(e.eid)
        for n, (_, ids) in enumerate(sorted(roots.items(), key=lambda kv: min(kv[1]))):
            out[(k, n)] = sorted(ids)
    return out


class TileSet:
    def __init__(self, model, exps, skies, cfg, log=print):
        """exps: the list of cuts (exps[i].cid == i); skies: {tile: SkyGrid}; model.sky must be one of them."""
        self.model, self.exps, self.cfg, self.log = model, exps, cfg, log
        self.dev = model.dev
        self.keys = sorted(skies)
        self.skies = skies
        self.multi = len(self.keys) > 1
        self.cuts = {k: [i for i, e in enumerate(exps) if e.tile == k] for k in self.keys}     # cut indices (positions in exps)
        self.nbytes = {k: sum(exps[c].nbytes() for c in self.cuts[k]) for k in self.keys}
        self.budget = float(getattr(cfg, "gpu_resident_gb", 4.0)) * 2 ** 30
        self.resident = OrderedDict()
        self.cur = None
        self.stats = {k: {"visits": 0, "steps": 0, "loads": 0, "load_seconds": 0.0} for k in self.keys}
        npx = {k: float(sum(int(exps[c].hostv("valid").sum()) for c in self.cuts[k] if exps[c].split == "train")) for k in self.keys}
        self.train_px = npx
        pw = float(getattr(cfg, "tile_weight_power", 0.5))
        w = np.array([npx[k] ** pw if npx[k] > 0 else 0.0 for k in self.keys])
        self.p = w / w.sum() if w.sum() > 0 else np.full(len(self.keys), 1.0 / len(self.keys))
        tr = [e for e in exps if e.split == "train"]
        self.shift_groups = [torch.as_tensor(v, device=self.dev, dtype=torch.long) for v in components(tr).values()]
        self.pedestal_groups = {k: torch.as_tensor(v, dtype=torch.long) for k, v in components(tr, key=lambda e: e.filt).items()}
        self.opt_sky, self.sig_b = {}, {}
        # --sky-swap: inactive skies (coefficients, buffers, Adam state) live on the host as well (a 0.02" sky of a 20.48" box with its
        # Adam state is ~0.3 GB, ten of them do not fit beside the data on an 11 GB slice)
        self.sky_swap = bool(getattr(cfg, "sky_swap", 0)) and self.multi
        if self.sky_swap:
            for k in self.keys:
                self._sky_to(k, torch.device("cpu"))
        # the model's current sky is the first tile's (and is active without a load when the data are already on the device)
        for k, s in skies.items():
            if s is model.sky:
                self.cur = k
                if not self.multi or exps[self.cuts[k][0]].host is None:
                    self.resident[k] = self.nbytes[k]
        if self.cur is not None and self.multi and (self.cur not in self.resident or self.sky_swap):
            self.cur = None

    def _sky_to(self, k, dev):
        """Move tile k's sky (parameters and buffers in place, so the optimiser keeps its keys) and its Adam state to dev."""
        sky = self.skies[k]
        for p in sky.parameters():
            p.grad = None
        sky.to(dev)
        opt = self.opt_sky.get(k)
        if opt is not None:
            for st in opt.state.values():
                for kk, v in list(st.items()):
                    if torch.is_tensor(v) and v.dim() > 0:
                        st[kk] = v.to(dev)

    @classmethod
    def single(cls, model, exps, cfg, log=print):
        tile = exps[0].tile if exps else 0
        return cls(model, exps, {tile: model.sky}, cfg, log)

    # ------------------------------------------------------------------ residency and the active tile
    def resident_gb(self):
        return sum(self.resident.values()) / 2 ** 30

    def activate(self, k):
        """Make tile k the model's tile: its sky becomes model.sky and its cuts' pixel tensors are on the device (LRU)."""
        if k == self.cur and (k in self.resident or not self.multi):
            if k in self.resident:
                self.resident.move_to_end(k)
            return False
        t0 = time.perf_counter()
        if k not in self.resident:
            need = self.nbytes[k]
            while self.resident and sum(self.resident.values()) + need > self.budget:
                old, _ = self.resident.popitem(last=False)
                for c in self.cuts[old]:
                    self.exps[c].release()
            for c in self.cuts[k]:
                self.exps[c].to_dev(self.dev)
            self.resident[k] = need
            self.stats[k]["loads"] += 1
        else:
            self.resident.move_to_end(k)
        if self.sky_swap:
            if self.cur is not None and self.cur != k:
                self._sky_to(self.cur, torch.device("cpu"))
            self._sky_to(k, self.dev)
        self.model.sky = self.skies[k]
        self.cur = k
        self.stats[k]["load_seconds"] += time.perf_counter() - t0
        return True

    def draw(self, rng):
        return self.keys[int(rng.choice(len(self.keys), p=self.p))]

    # ------------------------------------------------------------------ per-tile helpers
    def cut_ids(self, k, splits):
        return [c for c in self.cuts[k] if self.exps[c].split in splits]

    def make_sky_opts(self, cfg, sig_fn):
        """One Adam per tile sky (the single-box sky optimiser of make_optimizers, per tile)."""
        for k in self.keys:
            sky = self.skies[k]
            sb = sig_fn(sky)
            self.sig_b[k] = sb.to(self.dev)          # used on the device with the active sky (the sky itself may be on the host)
            lr = cfg.lr_sky * float(sb.mean())
            self.opt_sky[k] = torch.optim.Adam([{"params": list(sky.parameters()), "lr": lr, "base": lr}])

    def summary(self):
        return {"tiles": self.keys, "p": dict(zip(self.keys, [float(x) for x in self.p])), "train_px": self.train_px,
                "data_gb": {k: v / 2 ** 30 for k, v in self.nbytes.items()}, "stats": self.stats,
                "n_shift_components": len(self.shift_groups), "n_pedestal_groups": len(self.pedestal_groups),
                "resident_budget_gb": self.budget / 2 ** 30, "sky_swap": self.sky_swap}
