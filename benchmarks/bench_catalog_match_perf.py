#!/usr/bin/env python3
"""Fair crossmatch performance: torchsky vs scipy cKDTree (and astropy at small N).

Each (engine, workload) runs in its own subprocess, so peak RSS belongs to that
engine alone. Every engine gets the same thread count: ``torch.set_num_threads``
for torchsky and ``workers=`` for cKDTree. cKDTree pairs are flattened with
NumPy rather than a Python loop. Build and query are timed separately, and their
sum is the end-to-end time.

Pair-ID parity is checked before any timing is reported. Pairs on which the
engines disagree are recomputed in float64. Disagreements outside the 1e-8 arcsec
tie band around the radius mark the row ``FAIL``.

Layouts:

- ``sparse``: uniform all-sky, right = left jittered by 0.3 arcsec.
- ``dense``: a 100 deg^2 field, same jitter (the xmatcher audit setup).
- ``crowded``: Plummer clusters; right drawn independently.

Modes: ``best`` (nearest within radius), ``all`` (every pair) and ``knn2``
(two nearest, unbounded).

GPU nearest-neighbour peers (``--gpu-peers``) work on 3-D unit vectors with
chord radius ``2 sin(r/2)``. Inputs are uploaded to the device before timing (as
torchsky's ra/dec are) and results are copied back to the host inside the query
time. A peer whose import or CUDA check fails gives an ``UNAVAILABLE`` row.

- ``cupy_kdtree``: ``cupyx.scipy.spatial.KDTree`` in float64; ``best`` is
  ``query(k=1, distance_upper_bound=chord)``, ``knn2`` is ``query(k=2)`` and
  ``all`` is ``query_ball_point``, chunked because it allocates an
  ``(n_query, n_tree)`` int64 buffer (skipped above ``--cupy-ball-max-n``).
- ``cupy_kdtree_kgrow``: ``all`` only, via ``query(k=K, distance_upper_bound=chord)``;
  rows that fill all ``K`` slots are re-queried with ``2K`` until none does, so
  the pair set is exact.
- ``faiss_flat``: FAISS GPU ``IndexFlatL2`` (float32 only, kNN only).
- ``cuvs_brute_force``: cuVS ``brute_force`` (float32, kNN only).
- ``torch_cluster``: ``knn`` and ``radius`` on CUDA in float64; ``radius`` uses
  the same ``max_num_neighbors`` regrow so it never truncates.
- ``open3d_nns``: Open3D ``NearestNeighborSearch`` (``hybrid_search`` with
  ``max_knn=1``, ``fixed_radius_search``, ``knn_search``) in float64.

Brute-force peers (FAISS flat, cuVS brute force and torch_cluster, whose CUDA
kernels scan every reference point) are skipped above ``--brute-force-max-n``.
kNN-only peers have no ``all`` row. A peer whose pair IDs differ from torchsky
outside the tie band is ``INEXACT``, with its time kept but flagged; float32
peers are expected to be inexact at arcsecond scales. Without a torchsky
reference a peer row is ``UNVERIFIED``.

GPU memory: ``peak_gpu_bytes`` is the device-wide ``cudaMemGetInfo`` used
memory, sampled every millisecond during the warm-up call, minus the value
after CUDA context creation. It counts the inputs, the library's pools and
handles; short spikes between samples can be missed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR

TIE_BAND_ARCSEC = 1e-8

# engine -> (import probed before the layout is built, supported modes, precision)
GPU_PEERS: dict[str, tuple[str, frozenset[str], str]] = {
    "cupy_kdtree": ("cupyx.scipy.spatial", frozenset({"best", "all", "knn2"}), "float64"),
    "cupy_kdtree_kgrow": ("cupyx.scipy.spatial", frozenset({"all"}), "float64"),
    "faiss_flat": ("faiss", frozenset({"best", "knn2"}), "float32"),
    "cuvs_brute_force": ("cuvs.neighbors.brute_force", frozenset({"best", "knn2"}), "float32"),
    "torch_cluster": ("torch_cluster", frozenset({"best", "all", "knn2"}), "float64"),
    "open3d_nns": ("open3d.core", frozenset({"best", "all", "knn2"}), "float64"),
}
BRUTE_FORCE_PEERS = frozenset({"faiss_flat", "cuvs_brute_force", "torch_cluster"})
EXIT_UNAVAILABLE = 3
# Per-chunk scratch for chunked peer queries (bytes), capped at a quarter of free memory.
CHUNK_BUDGET_BYTES = 1 << 30


def _layout(name: str, n: int, seed: int) -> tuple[np.ndarray, ...]:
    g = np.random.default_rng(seed)
    if name == "sparse":
        ra = g.uniform(0, 360, n)
        dec = np.degrees(np.arcsin(g.uniform(-1, 1, n)))
    elif name == "dense":
        ra = 150.0 + g.uniform(0, 10, n)
        dec = 2.0 + g.uniform(0, 10, n)
    elif name == "crowded":
        centers = g.uniform([0, -60], [360, 60], size=(max(1, n // 20000), 2))
        k = g.integers(0, len(centers), n)
        u = g.uniform(1e-6, 1, n)
        r = np.minimum(30.0 / np.sqrt(u ** (-2 / 3) - 1), 1800.0) / 3600.0
        t = g.uniform(0, 2 * math.pi, n)
        dec = np.clip(centers[k, 1] + r * np.sin(t), -90, 90)
        ra = (centers[k, 0] + r * np.cos(t) / np.cos(np.radians(dec))) % 360
        g2 = np.random.default_rng(seed + 1)
        k2 = g2.integers(0, len(centers), n)
        u2 = g2.uniform(1e-6, 1, n)
        r2 = np.minimum(30.0 / np.sqrt(u2 ** (-2 / 3) - 1), 1800.0) / 3600.0
        t2 = g2.uniform(0, 2 * math.pi, n)
        rdec = np.clip(centers[k2, 1] + r2 * np.sin(t2), -90, 90)
        rra = (centers[k2, 0] + r2 * np.cos(t2) / np.cos(np.radians(rdec))) % 360
        return ra, dec, rra, rdec
    else:
        raise ValueError(name)
    j = np.random.default_rng(seed + 1)
    s = 0.3 / 3600.0
    rdec = np.clip(dec + j.normal(0, s, n), -90, 90)
    rra = (ra + j.normal(0, s, n) / np.maximum(np.cos(np.radians(dec)), 1e-9)) % 360
    return ra, dec, rra, rdec


def _unit(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], 1)


def _peak_rss_bytes() -> int:
    """This process's peak RSS.

    On Linux ``ru_maxrss`` survives fork + exec, so a worker would report its
    parent's peak; ``VmHWM`` belongs to the current address space and resets on exec.
    """
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


# ------------------------------------------------------------------ GPU peers


def _unavailable(reason: str) -> None:
    print(f"UNAVAILABLE: {reason}", file=sys.stderr)
    raise SystemExit(EXIT_UNAVAILABLE)


def _probe_peer(engine: str) -> None:
    """Import the peer and check it sees a CUDA device; exit 3 otherwise.

    Runs before the layout is built so a missing package costs nothing at 1e8.
    Open3D is imported before torch (its wheel bundles its own runtime libraries).
    """
    import importlib

    module = GPU_PEERS[engine][0]
    try:
        importlib.import_module(module)
    except Exception as exc:  # ImportError, or a CUDA library that fails to load
        _unavailable(f"{module}: {type(exc).__name__}: {exc}")
    import torch

    if not torch.cuda.is_available():
        _unavailable("torch.cuda.is_available() is False")
    try:
        if engine.startswith("cupy"):
            import cupy

            if cupy.cuda.runtime.getDeviceCount() < 1:
                _unavailable("cupy sees no CUDA device")
        elif engine == "faiss_flat":
            import faiss

            if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() < 1:
                _unavailable("faiss has no GPU support or sees no GPU")
        elif engine == "open3d_nns":
            import open3d.core as o3c

            if not o3c.cuda.is_available():
                _unavailable("open3d was built without CUDA or sees no GPU")
    except SystemExit:
        raise
    except Exception as exc:
        _unavailable(f"{engine} device probe: {type(exc).__name__}: {exc}")


class _GpuPeak:
    """Device-wide used memory (``cudaMemGetInfo``) above the post-init baseline."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self.total = int(total)
        self.base = int(total - free)
        self.peak = self.base
        self.name = torch.cuda.get_device_name()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            free, total = self._torch.cuda.mem_get_info()
            self.peak = max(self.peak, int(total - free))
            time.sleep(0.001)

    def __enter__(self) -> _GpuPeak:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        free, total = self._torch.cuda.mem_get_info()
        self.peak = max(self.peak, int(total - free))

    def record(self) -> dict[str, Any]:
        return {"peak_gpu_bytes": self.peak - self.base, "gpu_baseline_bytes": self.base,
                "gpu_total_bytes": self.total, "gpu_name": self.name,
                "gpu_mem_method": "cudaMemGetInfo sampled every 1 ms during warm-up, minus post-init baseline"}


def _chunk_rows(bytes_per_row: int) -> int:
    import torch

    free, _ = torch.cuda.mem_get_info()
    return max(1, min(CHUNK_BUDGET_BYTES, free // 4) // max(1, bytes_per_row))


def _gpu_peer(args: argparse.Namespace, A: np.ndarray, B: np.ndarray, chord: float):
    """Return (version, build, query) for a GPU peer; A queries, B is indexed."""
    import torch

    def sync() -> None:
        torch.cuda.synchronize()

    n = A.shape[0]
    engine, mode = args.engine, args.mode

    if engine in ("cupy_kdtree", "cupy_kdtree_kgrow"):
        import cupy as cp
        from cupyx.scipy.spatial import KDTree

        Ad, Bd = cp.asarray(A), cp.asarray(B)

        def build():
            tree = KDTree(Bd)
            sync()
            return tree

        def query(tree):
            if mode == "knn2":
                _, j = tree.query(Ad, k=2)
                return cp.asnumpy(j), None
            if mode == "best":
                _, j = tree.query(Ad, k=1, distance_upper_bound=chord)
                ok = j < tree.n
                return cp.asnumpy(cp.nonzero(ok)[0]), cp.asnumpy(j[ok])
            if engine == "cupy_kdtree":
                # query_ball_point fills an (n_query, n_tree) int64 buffer per call.
                rows = _chunk_rows(8 * int(tree.tree.shape[0]))
                lefts, rights = [], []
                for s in range(0, n, rows):
                    lists = list(tree.query_ball_point(Ad[s:s + rows], chord, return_sorted=False))
                    counts = np.fromiter((x.size for x in lists), dtype=np.int64, count=len(lists))
                    if counts.sum():
                        lefts.append(np.repeat(np.arange(s, s + len(lists)), counts))
                        rights.append(cp.asnumpy(cp.concatenate(lists)))
                if not lefts:
                    return np.empty(0, np.int64), np.empty(0, np.int64)
                return np.concatenate(lefts), np.concatenate(rights)
            # kgrow: k-NN bounded by the radius; rows that fill all K slots are redone with 2K.
            # A slot is a pair only if its index is valid and its distance within the chord
            # (missing neighbours are index tree.n); a row is saturated if its last slot is a pair.
            pending, k = cp.arange(n), 8
            lefts, rights = [], []
            while pending.size:
                if k > min(tree.n, 1 << 16):
                    d, j = tree.query(Ad[pending[:1]], k=min(k // 2, tree.n), distance_upper_bound=chord)
                    raise RuntimeError(f"k-regrow runaway: {int(pending.size)} rows still saturated at k={k // 2}; "
                                       f"row 0 d[:4]={cp.asnumpy(d[0, :4])} j[:4]={cp.asnumpy(j[0, :4])} "
                                       f"chord={chord:.3e} n={tree.n}")
                print(f"kgrow k={k} pending={int(pending.size)}", file=sys.stderr, flush=True)
                rows = _chunk_rows(16 * k)
                saturated = []
                for s in range(0, int(pending.size), rows):
                    r = pending[s:s + rows]
                    d, j = tree.query(Ad[r], k=k, distance_upper_bound=chord)
                    pair = (j < tree.n) & (d <= chord)
                    full = pair[:, -1]
                    ri, ci = cp.nonzero(pair & ~full[:, None])
                    lefts.append(r[ri])
                    rights.append(j[ri, ci])
                    saturated.append(r[full])
                pending, k = cp.concatenate(saturated), 2 * k
            return cp.asnumpy(cp.concatenate(lefts)), cp.asnumpy(cp.concatenate(rights))

        return cp.__version__, build, query

    if engine == "faiss_flat":
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401  (torch CUDA tensors in/out)

        res = faiss.StandardGpuResources()
        X = torch.from_numpy(B).float().cuda()
        Y = torch.from_numpy(A).float().cuda()
        chord2 = float(np.float32(chord) ** 2)

        def build():
            index = faiss.GpuIndexFlatL2(res, 3)
            index.add(X)
            sync()
            return index

        def query(index):
            d, j = index.search(Y, 2 if mode == "knn2" else 1)
            if mode == "knn2":
                return j.cpu().numpy(), None
            ok = (j[:, 0] >= 0) & (d[:, 0] <= chord2)
            return torch.nonzero(ok)[:, 0].cpu().numpy(), j[ok, 0].cpu().numpy()

        return faiss.__version__, build, query

    if engine == "cuvs_brute_force":
        import cuvs
        from cuvs.neighbors import brute_force

        X = torch.from_numpy(B).float().cuda()
        Y = torch.from_numpy(A).float().cuda()
        chord2 = float(np.float32(chord) ** 2)

        def build():
            index = brute_force.build(X, metric="sqeuclidean")
            sync()
            return index

        def query(index):
            d, j = brute_force.search(index, Y, 2 if mode == "knn2" else 1)
            d = torch.as_tensor(d, device="cuda")
            j = torch.as_tensor(j, device="cuda").long()
            if mode == "knn2":
                return j.cpu().numpy(), None
            ok = (j[:, 0] >= 0) & (d[:, 0] <= chord2)
            return torch.nonzero(ok)[:, 0].cpu().numpy(), j[ok, 0].cpu().numpy()

        return cuvs.__version__, build, query

    if engine == "torch_cluster":
        import torch_cluster

        X = torch.from_numpy(B).cuda()
        Y = torch.from_numpy(A).cuda()

        def build():
            return None  # no index: the CUDA kernels scan X per query

        def query(_):
            if mode in ("best", "knn2"):
                k = 2 if mode == "knn2" else 1
                e = torch_cluster.knn(X, Y, k)  # e[0] indexes Y, e[1] indexes X
                if mode == "knn2":
                    order = torch.argsort(e[0], stable=True)
                    return e[1][order].view(-1, 2).cpu().numpy(), None
                d2 = ((Y[e[0]] - X[e[1]]) ** 2).sum(1)
                ok = d2 <= chord * chord
                return e[0][ok].cpu().numpy(), e[1][ok].cpu().numpy()
            # radius keeps at most max_num_neighbors per row: regrow rows that hit the cap.
            pending, k = torch.arange(n, device="cuda"), 8
            lefts, rights = [], []
            while pending.numel():
                e = torch_cluster.radius(X, Y[pending], chord, max_num_neighbors=k)
                counts = torch.bincount(e[0], minlength=pending.numel())
                full = counts >= k
                keep = ~full[e[0]]
                lefts.append(pending[e[0][keep]])
                rights.append(e[1][keep])
                pending, k = pending[full], 2 * k
            return torch.cat(lefts).cpu().numpy(), torch.cat(rights).cpu().numpy()

        return getattr(torch_cluster, "__version__", "unknown"), build, query

    if engine == "open3d_nns":
        import open3d
        import open3d.core as o3c

        dev = o3c.Device("CUDA:0")
        X = o3c.Tensor(np.ascontiguousarray(B), device=dev)
        Y = o3c.Tensor(np.ascontiguousarray(A), device=dev)

        def build():
            nns = o3c.nns.NearestNeighborSearch(X)
            radius_index = {"best": nns.hybrid_index, "all": nns.fixed_radius_index}.get(mode)
            built = radius_index(chord) if radius_index is not None else nns.knn_index()
            if built is False:
                raise RuntimeError(f"open3d index build failed for mode {mode}")
            sync()
            return nns

        def query(nns):
            if mode == "knn2":
                j, _ = nns.knn_search(Y, 2)
                return j.cpu().numpy().astype(np.int64), None
            if mode == "best":
                j, _, _ = nns.hybrid_search(Y, chord, 1)
                j = j.cpu().numpy().reshape(-1).astype(np.int64)
                ok = j >= 0
                return np.nonzero(ok)[0], j[ok]
            j, _, splits = nns.fixed_radius_search(Y, chord, False)
            counts = np.diff(splits.cpu().numpy().astype(np.int64))
            return np.repeat(np.arange(n), counts), j.cpu().numpy().astype(np.int64)

        return open3d.__version__, build, query

    raise ValueError(engine)


# ------------------------------------------------------------------ workers


def _worker(args: argparse.Namespace) -> None:
    peer = args.engine in GPU_PEERS
    if peer:
        _probe_peer(args.engine)
    la, lb, ra, rb = _layout(args.layout, args.n, args.seed)
    out = Path(args.out)
    t_build = 0.0
    gpu = _GpuPeak() if peer or (args.engine == "torchsky" and args.device.startswith("cuda")) else None
    extra: dict[str, Any] = {}
    if peer:
        chord = 2 * math.sin(math.radians(args.radius / 3600) / 2)
        version, build, query = _gpu_peer(args, _unit(la, lb), _unit(ra, rb), chord)
        extra["precision"] = GPU_PEERS[args.engine][2]
    elif args.engine == "torchsky":
        import torch

        import torchsky
        from torchsky.catalogs import SkyIndex, crossmatch_sky

        torch.set_num_threads(args.threads)
        dev = torch.device(args.device)
        L = (torch.from_numpy(la).to(dev), torch.from_numpy(lb).to(dev))
        R = (torch.from_numpy(ra).to(dev), torch.from_numpy(rb).to(dev))

        def sync() -> None:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        def build():
            index = SkyIndex.build(*R) if args.mode == "knn2" else None
            sync()
            return index

        def query(index):
            if args.mode == "knn2":
                res = index.query_nearest(*L, 2)
                sync()
                return res.index.cpu().numpy(), None
            res = crossmatch_sky(*L, *R, radius_arcsec=args.radius, find=args.mode)
            sync()
            return res.left_index.cpu().numpy(), res.right_index.cpu().numpy()

        version = torchsky.__version__
    elif args.engine == "ckdtree":
        import scipy
        from scipy.spatial import cKDTree

        A, B = _unit(la, lb), _unit(ra, rb)
        chord = 2 * math.sin(math.radians(args.radius / 3600) / 2)

        def build():
            return cKDTree(B)

        def query(tree):
            if args.mode == "knn2":
                _, j = tree.query(A, k=2, workers=args.threads)
                return j, None
            if args.mode == "best":
                d, j = tree.query(A, k=1, distance_upper_bound=chord, workers=args.threads)
                ok = np.isfinite(d)
                return np.nonzero(ok)[0], j[ok]
            lists = tree.query_ball_point(A, chord, workers=args.threads, return_sorted=False)
            counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
            flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]) if counts.sum() else np.empty(0, np.int64)
            return np.repeat(np.arange(len(lists)), counts), flat

        version = scipy.__version__
    elif args.engine == "astropy":
        import astropy
        import astropy.units as u
        from astropy.coordinates import SkyCoord, search_around_sky

        def build():
            return SkyCoord(ra * u.deg, rb * u.deg)

        left = SkyCoord(la * u.deg, lb * u.deg)

        def query(right):
            if args.mode == "knn2":
                j2 = left.match_to_catalog_sky(right, nthneighbor=2)[0]
                j1 = left.match_to_catalog_sky(right, nthneighbor=1)[0]
                return np.stack([j1, j2], 1), None
            if args.mode == "best":
                j, d, _ = left.match_to_catalog_sky(right)
                ok = d.arcsec <= args.radius
                return np.nonzero(ok)[0], j[ok]
            il, ir, _, _ = search_around_sky(left, right, args.radius * u.arcsec)
            return il, ir

        version = astropy.__version__
    else:
        raise ValueError(args.engine)

    samples = []
    if gpu is not None and args.engine == "torchsky":
        import torch

        torch.cuda.reset_peak_memory_stats()
    for rep in range(args.runs + 1):  # first call warms up; GPU memory is sampled there
        with gpu if (gpu is not None and rep == 0) else contextlib.nullcontext():
            t0 = time.perf_counter()
            index = build()
            t1 = time.perf_counter()
            li, rj = query(index)
            t2 = time.perf_counter()
        if rep:
            samples.append((t1 - t0, t2 - t1))
        del index
    if gpu is not None:
        extra.update(gpu.record())
        if args.engine == "torchsky":
            extra["peak_gpu_torch_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    t_build = float(np.median([s[0] for s in samples]))
    t_query = float(np.median([s[1] for s in samples]))
    np.save(out / f"{args.engine}_left.npy", np.asarray(li))
    if rj is not None:
        np.save(out / f"{args.engine}_right.npy", np.asarray(rj))
    (out / f"{args.engine}.json").write_text(json.dumps({
        "engine": args.engine, "engine_version": version, "build_s": t_build,
        "query_s": t_query, "time_s": t_build + t_query, "peak_rss_bytes": _peak_rss_bytes(),
        "pairs": int(np.asarray(li).shape[0]) if rj is not None else int(np.asarray(li).shape[0]),
        **extra,
    }))


# ------------------------------------------------------------------ parent


def _sep_arcsec(la, lb, ra, rb, li, rj) -> np.ndarray:
    a, b = _unit(la[li], lb[li]), _unit(ra[rj], rb[rj])
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))) * 3600


def _parity(args, workload_dir: Path, engines: list[str]) -> dict[str, Any]:
    """Compare every engine with ``engines[0]``; vectorised so inexact peers stay cheap."""
    la, lb, ra, rb = _layout(args.layout, args.n, args.seed)
    ref = engines[0]
    out: dict[str, Any] = {}
    if args.mode == "knn2":
        a = np.load(workload_dir / f"{ref}_left.npy").astype(np.int64)
        for other in engines[1:]:
            b = np.load(workload_dir / f"{other}_left.npy").astype(np.int64)
            if b.shape != a.shape:
                out[other] = {"differing_rows": int(a.shape[0]), "mismatches": int(a.shape[0]),
                              "shape": list(b.shape)}
                continue
            rows = np.nonzero((a != b).any(axis=1))[0]
            invalid = ((b[rows] < 0) | (b[rows] >= args.n)).any(axis=1)
            good = rows[~invalid]
            # float64 near-ties between neighbours are ambiguous: compare sorted separations.
            sa = _sep_arcsec(la, lb, ra, rb, np.repeat(good, 2), a[good].ravel()).reshape(-1, 2)
            sb = _sep_arcsec(la, lb, ra, rb, np.repeat(good, 2), b[good].ravel()).reshape(-1, 2)
            off = np.abs(np.sort(sa, axis=1) - np.sort(sb, axis=1)).max(axis=1, initial=0.0)
            bad = int(invalid.sum()) + int(np.count_nonzero(off > TIE_BAND_ARCSEC))
            out[other] = {"differing_rows": int(rows.size), "mismatches": bad}
        return out

    def key(engine: str) -> tuple[np.ndarray, int]:
        left = np.load(workload_dir / f"{engine}_left.npy").astype(np.int64)
        right = np.load(workload_dir / f"{engine}_right.npy").astype(np.int64)
        valid = (left >= 0) & (left < args.n) & (right >= 0) & (right < args.n)
        return left[valid] * args.n + right[valid], int(np.count_nonzero(~valid))

    base, _ = key(ref)
    for other in engines[1:]:
        k, invalid = key(other)
        diff = np.setxor1d(base, k)  # sorted, so li is sorted
        li, rj = diff // args.n, diff % args.n
        sep = _sep_arcsec(la, lb, ra, rb, li, rj)
        if args.mode == "best":
            # A different nearest partner is fine only if equidistant within the band.
            _, first, counts = np.unique(li, return_index=True, return_counts=True)
            second = np.minimum(first + 1, max(li.size - 1, 0))
            tie = (counts == 2) & (np.abs(sep[first] - sep[second]) <= TIE_BAND_ARCSEC)
            bad = int(np.count_nonzero(~tie))
        else:
            bad = int(np.count_nonzero(np.abs(sep - args.radius) > TIE_BAND_ARCSEC))
        out[other] = {"differing_pairs": int(diff.size), "mismatches": bad + invalid}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=lambda s: [int(float(x)) for x in s.split(",")], default=[10_000, 100_000, 1_000_000])
    p.add_argument("--layouts", type=lambda s: s.split(","), default=["sparse", "dense", "crowded"])
    p.add_argument("--modes", type=lambda s: s.split(","), default=["best", "all", "knn2"])
    p.add_argument("--threads", type=lambda s: [int(x) for x in s.split(",")], default=[os.cpu_count() or 1])
    p.add_argument("--radius-arcsec", type=float, default=2.0)
    p.add_argument("--device", default="cpu", help="torchsky device (cKDTree and astropy run on CPU)")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--astropy-max-n", type=int, default=100_000)
    p.add_argument("--min-ratio-vs-ckdtree", type=float, default=None,
                   help="fail rows where cKDTree time / torchsky time is below this")
    p.add_argument("--min-ratio-size", type=int, default=1_000_000,
                   help="apply --min-ratio-vs-ckdtree only from this size up")
    p.add_argument("--max-rss-ratio-vs-ckdtree", type=float, default=None)
    p.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "catalog_match_perf.json")
    p.add_argument("--gpu-peers", type=lambda s: [] if s in ("", "none") else s.split(","),
                   default=list(GPU_PEERS), help="GPU nearest-neighbour peers to run ('none' to skip)")
    p.add_argument("--brute-force-max-n", type=int, default=1_000_000,
                   help="skip brute-force GPU peers (FAISS flat, cuVS, torch_cluster) above this size")
    p.add_argument("--cupy-ball-max-n", type=int, default=1_000_000,
                   help="skip cupy query_ball_point (O(n_query * n_tree) buffer) above this size")
    p.add_argument("--worker-timeout", type=float, default=None, help="seconds per engine subprocess")
    # internal worker arguments
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--engine", help=argparse.SUPPRESS)
    p.add_argument("--layout", help=argparse.SUPPRESS)
    p.add_argument("--mode", help=argparse.SUPPRESS)
    p.add_argument("--n", type=int, help=argparse.SUPPRESS)
    p.add_argument("--radius", type=float, help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        args.threads = args.threads[0]
        _worker(args)
        return 0
    unknown = sorted(set(args.gpu_peers) - set(GPU_PEERS))
    if unknown:
        p.error(f"unknown --gpu-peers {unknown}; choose from {sorted(GPU_PEERS)}")

    def skip_reason(engine: str, mode: str, n: int) -> str | None:
        if engine not in GPU_PEERS:
            return None
        if mode not in GPU_PEERS[engine][1]:
            if engine == "cupy_kdtree_kgrow":
                return "variant exists only for mode 'all'"
            return f"kNN-only API: no exact radius query for mode '{mode}'"
        if engine in BRUTE_FORCE_PEERS and n > args.brute_force_max_n:
            return f"brute-force O(N*M) GPU kernel; skipped above n={args.brute_force_max_n}"
        if engine == "cupy_kdtree" and mode == "all" and n > args.cupy_ball_max_n:
            return (f"query_ball_point allocates an (n_query, n_tree) int64 buffer (O(N^2) memory/fill); "
                    f"skipped above n={args.cupy_ball_max_n}; see cupy_kdtree_kgrow")
        return None

    unavailable: dict[str, list[str]] = {}  # peers that failed their probe once are not retried
    rows: list[dict[str, Any]] = []
    for threads in args.threads:
        for layout in args.layouts:
            for n in args.sizes:
                for mode in args.modes:
                    engines = ["torchsky", "ckdtree"] + (["astropy"] if n <= args.astropy_max_n else []) \
                        + list(args.gpu_peers)
                    with tempfile.TemporaryDirectory(prefix="xm_perf_") as tmp:
                        results: dict[str, dict[str, Any]] = {}
                        for engine in engines:
                            reason = skip_reason(engine, mode, n)
                            if reason is not None:
                                status = "UNSUPPORTED" if mode not in GPU_PEERS[engine][1] else "SKIPPED"
                                results[engine] = {"skip": status, "error": [reason]}
                                continue
                            if engine in unavailable:
                                results[engine] = {"unavailable": True, "error": unavailable[engine]}
                                continue
                            cmd = [sys.executable, "-m", "torchsky_harness.benchmarks.bench_catalog_match_perf",
                                   "--worker", "--engine", engine, "--layout", layout, "--mode", mode,
                                   "--n", str(n), "--radius", str(args.radius_arcsec), "--seed", str(args.seed),
                                   "--runs", str(args.runs), "--threads", str(threads), "--out", tmp,
                                   "--device", args.device]
                            env = dict(os.environ, OMP_NUM_THREADS=str(threads))
                            try:
                                proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                                      timeout=args.worker_timeout)
                            except subprocess.TimeoutExpired as exc:
                                tail = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                                results[engine] = {"timeout": True, "error": [f"exceeded --worker-timeout {args.worker_timeout}s"],
                                                   "stderr_tail": tail.strip().splitlines()[-15:]}
                                continue
                            if proc.returncode == EXIT_UNAVAILABLE and engine in GPU_PEERS:
                                unavailable[engine] = proc.stderr.strip().splitlines()[-1:] or ["unavailable"]
                                results[engine] = {"unavailable": True, "error": unavailable[engine]}
                            elif proc.returncode != 0:
                                results[engine] = {"error": proc.stderr.strip().splitlines()[-1:] or ["failed"],
                                                   "stderr_tail": proc.stderr.strip().splitlines()[-15:]}
                            else:
                                results[engine] = json.loads((Path(tmp) / f"{engine}.json").read_text())
                        ok_engines = [e for e in engines if "error" not in results[e]]
                        parity = _parity(argparse.Namespace(layout=layout, n=n, seed=args.seed, mode=mode,
                                                            radius=args.radius_arcsec), Path(tmp), ok_engines) \
                            if "torchsky" in ok_engines and len(ok_engines) > 1 else {}
                    ts = results.get("torchsky", {})
                    for engine in engines:
                        r = results[engine]
                        peer = engine in GPU_PEERS
                        row: dict[str, Any] = {"case": f"{layout}_{mode}_n{n}_t{threads}", "layout": layout,
                                               "mode": mode, "n_left": n, "n_right": n, "threads": threads,
                                               "radius_arcsec": args.radius_arcsec, "library": engine,
                                               "device": "cuda" if peer else args.device if engine == "torchsky" else "cpu"}
                        if peer:
                            row["precision"] = GPU_PEERS[engine][2]
                        if "error" in r:
                            if engine == "torchsky":
                                status = "FAIL"
                            elif "skip" in r:
                                status = r["skip"]
                            elif "unavailable" in r or not peer:
                                status = "UNAVAILABLE"
                            else:
                                status = "TIMEOUT" if "timeout" in r else "ERROR"
                            row.update(status=status, reason=r["error"])
                            if "stderr_tail" in r:
                                row["stderr_tail"] = r["stderr_tail"]
                        else:
                            row.update(r)
                            row["rows_per_s"] = n / r["time_s"]
                            row["pairs_per_s"] = r["pairs"] / r["time_s"]
                            row["candidate_reduction"] = "unavailable: subject exposes no public candidate counters"
                            row["status"] = "OK"
                        if engine != "torchsky" and "error" not in r and "time_s" not in ts and peer:
                            row["status"] = "UNVERIFIED"
                            row["reason"] = "no torchsky reference for pair-ID parity"
                        if engine != "torchsky" and "error" not in r and "time_s" in ts:
                            row["speed_ratio_torchsky_vs_this"] = r["time_s"] / ts["time_s"]
                            row["rss_ratio_torchsky_vs_this"] = ts["peak_rss_bytes"] / r["peak_rss_bytes"]
                            if peer and ts.get("peak_gpu_bytes") and r.get("peak_gpu_bytes"):
                                row["gpu_mem_ratio_torchsky_vs_this"] = ts["peak_gpu_bytes"] / r["peak_gpu_bytes"]
                            row.update({f"parity_{k}": v for k, v in parity.get(engine, {}).items()})
                            if parity.get(engine, {}).get("mismatches", 0):
                                if peer:
                                    row["status"] = "INEXACT"
                                    row["reason"] = ("pair-ID parity mismatch outside the tie band; time is for a wrong result"
                                                     + (" (float32 distances)" if GPU_PEERS[engine][2] == "float32" else ""))
                                else:
                                    row["status"] = "FAIL"
                                    row["reason"] = "pair-ID parity mismatch outside the tie band"
                            if engine == "ckdtree" and n >= args.min_ratio_size:
                                if args.min_ratio_vs_ckdtree is not None and row["speed_ratio_torchsky_vs_this"] < args.min_ratio_vs_ckdtree:
                                    row["status"] = "FAIL"
                                    row["reason"] = f"speed ratio below {args.min_ratio_vs_ckdtree}"
                                if args.max_rss_ratio_vs_ckdtree is not None and row["rss_ratio_torchsky_vs_this"] > args.max_rss_ratio_vs_ckdtree:
                                    row["status"] = "FAIL"
                                    row["reason"] = f"RSS ratio above {args.max_rss_ratio_vs_ckdtree}"
                        rows.append(row)
                        if "error" in r:
                            print(f"{row['case']:<28} {engine:<17} {row['status']} {row['reason']}", flush=True)
                            continue
                        gpu = f"gpu={row['peak_gpu_bytes']/2**20:.0f}MiB " if "peak_gpu_bytes" in row else ""
                        print(f"{row['case']:<28} {engine:<17} {row.get('status')} "
                              f"t={row.get('time_s', float('nan')):.3f}s (build {row.get('build_s', float('nan')):.3f}) "
                              f"rss={row.get('peak_rss_bytes', 0)/2**20:.0f}MiB {gpu}"
                              f"ratio={row.get('speed_ratio_torchsky_vs_this', float('nan')):.2f} "
                              f"parity={ {k: v for k, v in row.items() if k.startswith('parity_')} }", flush=True)
                    # Written after every case so a job killed mid-run keeps what it measured.
                    args.json_out.parent.mkdir(parents=True, exist_ok=True)
                    args.json_out.write_text(json.dumps(rows, indent=2))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))
    failed = [r for r in rows if r.get("status") == "FAIL"]
    print(f"{len(failed)} failing row(s). JSON: {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
