"""Backend-matrix adapter: exclusive-association belief-propagation fixed-point solve.

Exercises ``torchsky.catalogs._exclusive._Graph.solve`` for the pairwise model (one
secondary catalogue), the solve that ``pairwise_exclusive`` / ``associate(...,
exclusive=True)`` run. The candidate graph of a crowded synthetic scene and its
contested subgraph (``_prepare``) are built once in ``make_inputs``; only the solve
(Anderson-accelerated message sweeps to ``max |dx| <= tol``) is timed.

Native gates in ``_exclusive`` (all decided when the graph is built, by ``_prepare``):

- ``_Graph.__init__``: on CPU, pairwise layout, ``_zones._native_ext()`` not None and
  ``hasattr(ext, "xmatch_bp_pairwise_solve")`` -> ``graph.native`` is set and
  ``solve`` runs the whole fixed point in ``xmatch_bp_pairwise_solve``.
- ``_Segments.__init__`` (groups and sources): on CPU with ``hasattr(ext,
  "xmatch_segment_lse_excl")`` -> the torch sweep's two segment reductions call that
  kernel (when no grad is needed).

Both read ``_zones._native_ext`` at build time, so the pure-torch graph is built with
it patched to return None (``graph.native is None`` and ``groups.ext is
sources.ext is None``), and the call also runs under the patch with ``native=False``.
"""

from __future__ import annotations

import contextlib
import math
import types
from collections.abc import Callable, Iterator
from typing import Any

import torch

from torchsky.catalogs import _exclusive, _zones

SIZES = {"small": 5_000, "medium": 50_000, "large": 200_000}
DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
_SEEDS = {"small": 32_001, "medium": 32_002, "large": 32_003}

# Crowded field: primary density 0.02 arcsec^-2 (72k per deg^2); 60% of primaries
# have a counterpart (1 arcsec Gaussian offset per axis); background secondaries at
# 0.4x the primary density. Candidates within 5 arcsec: ~1.6 chance + the true one,
# so most secondaries are claimed by several primaries (a loopy contested graph).
_DENSITY = 0.02
_COUNTERPART_FRAC = 0.6
_BG_FRAC = 0.4
_SIGMA = 1.0
_RADIUS = 5.0
_RA0, _DEC0 = 150.0, 2.0

_TOL = {torch.float64: 1e-9, torch.float32: 1e-5}  # library default 1e-9 (float64)
_MAX_ITER = 2000

# Posterior / message agreement. float64: both paths run the same Anderson iteration
# to max |dx| <= 1e-9 in log-message units; the remaining distance to the fixed
# point is ~tol / (1 - contraction), so 1e-6 absolute on probabilities is generous.
# float32: tol 1e-5 (float32 eps ~1.2e-7 times log-message magnitudes up to ~30
# makes 1e-9 unreachable), so allow 2e-3 absolute on probabilities.
_ATOL_F64 = 1e-6
_ATOL_F32 = 2e-3

NOTES = (
    "Times _exclusive._Graph.solve(x0=0, logw, tol, max_iter=2000) for the pairwise "
    "exclusive model on a crowded scene (primary density 0.02 arcsec^-2, 60% with a "
    "1-arcsec counterpart, background secondaries at 0.4x, candidates within 5 arcsec, "
    "log w = log pi + Gaussian log LR). Candidate pairs (CPU float64 zone search) and "
    "_prepare (contested subgraph) are built once in make_inputs. Native gates: "
    "_Graph.__init__ sets graph.native (xmatch_bp_pairwise_solve) and _Segments.__init__ "
    "sets ext (xmatch_segment_lse_excl) on CPU when _zones._native_ext() has those "
    "attributes; the pure-torch graph is built and called with _zones._native_ext patched to "
    "None and solve(native=False), so eager runs _anderson + torch_sweep only (smoke run "
    "counts 0 _cpp_ext.xmatch_* calls). 'native' (CPU float64) is a second graph built "
    "unpatched; one xmatch_bp_pairwise_solve call. _exclusive hard-codes float64 (_F64), "
    "which MPS lacks: on MPS the adapter patches _exclusive._F64 = float32 for build and "
    "call (same code, float32 arithmetic, tol 1e-5, strict=False); this is not a supported "
    "torchsky configuration. 'compile' is torch.compile(solve) with default dynamic shapes: "
    "_anderson syncs to the host every sweep (float(res) for the convergence test, restart "
    "logic, Python ring buffer), so Dynamo graph-breaks per iteration and effectively compiles "
    "the sweep (torch_sweep: gathers, index_copy, scatter_reduce/index_add segment LSE) and "
    "the Anderson mixing step separately; 'compile_fullgraph' (fullgraph=True on the whole "
    "solve) is expected to raise. Output: log messages, sweep count, and posteriors "
    "recomputed in check(); check() compares posteriors and exp(log_rho) with atol 1e-6 "
    "(float64) / 2e-3 (float32). MPS on macOS < 15 raises in eager and compile: "
    "lse_excluding_self calls int64 Tensor.scatter_reduce('amin') directly instead of "
    "torchsky.device._scatter_reduce_int (a torchsky gap, not patched here); with that "
    "routed through _scatter_reduce_int the float32 MPS solve matched CPU native to 5e-7. "
    "Compiled variants use a separate code object each (_fresh) so the fullgraph compile "
    "does not reuse the default compile's Dynamo cache entry."
)


def _no_native() -> None:
    return None


def _fresh(fn: Callable, tag: str) -> Callable:
    """Copy of ``fn`` with its own code object.

    Dynamo caches compiled frames on the code object, so ``torch.compile(fn,
    fullgraph=True)`` after ``torch.compile(fn)`` silently reuses the graph-broken
    entry and never raises (torch 2.14.1). Each compiled variant gets its own copy.
    """
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return types.FunctionType(
        code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
    )


@contextlib.contextmanager
def force_pure_torch(dtype: torch.dtype = torch.float64) -> Iterator[None]:
    """No native kernels inside the block; ``_exclusive`` works in ``dtype``."""
    orig_ext, orig_f64 = _zones._native_ext, _exclusive._F64
    _zones._native_ext = _no_native
    _exclusive._F64 = dtype
    try:
        yield
    finally:
        _zones._native_ext = orig_ext
        _exclusive._F64 = orig_f64


def _offset(ra, dec, dx_arcsec, dy_arcsec):
    dec2 = dec + dy_arcsec / 3600.0
    ra2 = ra + dx_arcsec / 3600.0 / torch.cos(torch.deg2rad(dec))
    return ra2, dec2


def _scene(n: int, seed: int):
    """``(li, rj, log_w, log_a)`` candidate pairs sorted by primary (CPU float64)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    side = math.sqrt(n / _DENSITY) / 3600.0  # deg; flat-sky box is fine at < 2 deg
    cosd = math.cos(math.radians(_DEC0))
    lra = _RA0 + torch.rand(n, generator=gen, dtype=torch.float64) * side / cosd
    ldec = _DEC0 + torch.rand(n, generator=gen, dtype=torch.float64) * side
    m = int(n * _COUNTERPART_FRAC)
    src = torch.randperm(n, generator=gen)[:m]
    z = torch.randn(m, 2, generator=gen, dtype=torch.float64) * _SIGMA
    cra, cdec = _offset(lra[src], ldec[src], z[:, 0], z[:, 1])
    nb = int(n * _BG_FRAC)
    bra = _RA0 + torch.rand(nb, generator=gen, dtype=torch.float64) * side / cosd
    bdec = _DEC0 + torch.rand(nb, generator=gen, dtype=torch.float64) * side
    rra, rdec = torch.cat((cra, bra)), torch.cat((cdec, bdec))
    perm = torch.randperm(rra.numel(), generator=gen)
    rra, rdec = rra[perm], rdec[perm]
    index = _zones.ZoneIndex(rra, rdec, zone_height_deg=_RADIUS / 3600.0)
    radius = torch.full((n,), _RADIUS, dtype=torch.float64)
    li, rj, sep = _zones.zone_query(index, lra, ldec, radius, find="all")
    # log w = log(pi) + log LR, LR = N(sep; sigma_2d) / background secondary density.
    pi = _COUNTERPART_FRAC
    rho_bg = (_COUNTERPART_FRAC + _BG_FRAC) * _DENSITY  # secondaries per arcsec^2
    # Only the secondary is scattered: offset variance sigma^2 per axis.
    two_s2 = 2.0 * _SIGMA * _SIGMA
    log_lr = -(sep * sep) / two_s2 - math.log(math.pi * two_s2) - math.log(rho_bg)
    log_w = math.log(pi) + log_lr
    log_a = torch.full((n,), math.log(1.0 - pi), dtype=torch.float64)
    return li, rj, log_w, log_a


def _graph(li, rj, log_w, log_a, device, dtype):
    n_left, k = int(log_a.numel()), int(li.numel())
    owner = torch.cat((li, torch.arange(n_left))).to(device)
    source = torch.cat((rj, torch.full((n_left,), -1, dtype=torch.int64))).to(device)
    slot = torch.arange(k + n_left, device=device)[:, None]
    graph = _exclusive._prepare(owner, slot, owner, owner, source)
    if graph is None:
        raise RuntimeError("scene has no contested secondaries")
    logw = torch.cat((log_w, log_a)).to(dtype).to(device)[graph.hyp]
    x0 = torch.zeros(int(graph.msg_idx.numel()), dtype=dtype, device=device)
    return graph, logw, x0


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    device = torch.device(device)
    n = SIZES[size_key]
    li, rj, log_w, log_a = _scene(n, _SEEDS[size_key])
    with force_pure_torch(dtype):
        graph, logw, x0 = _graph(li, rj, log_w, log_a, device, dtype)
    assert (
        graph.native is None and graph.groups.ext is None and graph.sources.ext is None
    )
    out = {
        "graph": graph,
        "logw": logw,
        "x0": x0,
        "dtype": dtype,
        "tol": _TOL.get(dtype, 1e-9),
        "strict": dtype == torch.float64,
        # CPU float64 bookkeeping for posteriors in check().
        "aux": {
            "li": li,
            "log_w": log_w,
            "log_a": log_a,
            "msg_slot": graph.slots[graph.msg_idx].to("cpu"),
        },
    }
    if device.type == "cpu" and dtype == torch.float64 and _native_available():
        ngraph, nlogw, nx0 = _graph(li, rj, log_w, log_a, device, dtype)
        assert ngraph.native is not None
        assert torch.equal(ngraph.slots[ngraph.msg_idx], out["aux"]["msg_slot"])
        out.update(native_graph=ngraph, native_logw=nlogw, native_x0=nx0)
    return out


def _native_available() -> bool:
    ext = _zones._native_ext()
    return ext is not None and hasattr(ext, "xmatch_bp_pairwise_solve")


def items(size_key: str) -> int:
    return SIZES[size_key]


def _solve(graph, x0, logw, tol: float, strict: bool) -> torch.Tensor:
    return graph.solve(x0, logw, tol, _MAX_ITER, strict, native=False)


def _torch_variant(solve: Callable[..., torch.Tensor]) -> Callable[[dict], dict]:
    def run(inputs: dict) -> dict:
        g = inputs["graph"]
        with torch.no_grad(), force_pure_torch(inputs["dtype"]):
            x = solve(g, inputs["x0"], inputs["logw"], inputs["tol"], inputs["strict"])
        return {"log_rho": x, "sweeps": g.last_sweeps, "aux": inputs["aux"]}

    return run


def _native(inputs: dict) -> dict:
    g = inputs["native_graph"]
    with torch.no_grad():
        x = g.solve(inputs["native_x0"], inputs["native_logw"], 1e-9, _MAX_ITER, True)
    return {"log_rho": x, "sweeps": g.last_sweeps, "aux": inputs["aux"]}


def variants(device: torch.device) -> dict[str, Callable[[dict], Any]]:
    device = torch.device(device)
    out: dict[str, Callable[[dict], Any]] = {
        "eager": _torch_variant(_solve),
        "compile": _torch_variant(torch.compile(_fresh(_solve, "compile"))),
        "compile_fullgraph": _torch_variant(
            torch.compile(_fresh(_solve, "fullgraph"), fullgraph=True)
        ),
    }
    if device.type == "cpu" and _native_available():
        out["native"] = _native
    return out


def _posterior(res: dict) -> tuple[torch.Tensor, torch.Tensor]:
    aux = res["aux"]
    li, log_w, log_a = aux["li"], aux["log_w"], aux["log_a"]
    k, n_left = int(li.numel()), int(log_a.numel())
    full = torch.zeros(k + n_left, dtype=torch.float64)
    full[aux["msg_slot"]] = res["log_rho"].detach().to("cpu").to(torch.float64)
    rho = full[:k]
    owner = torch.cat((li, torch.arange(n_left)))
    u = torch.cat((log_w + rho, log_a))
    z = _exclusive._segment_lse(u, owner, n_left)
    return torch.exp(u[:k] - z[li]), torch.exp(rho)


def check(reference: dict, candidate: dict) -> tuple[bool, str]:
    rl, cl = reference["log_rho"], candidate["log_rho"]
    if rl.shape != cl.shape:
        return False, f"shape mismatch {tuple(rl.shape)} vs {tuple(cl.shape)}"
    if not bool(torch.isfinite(cl.detach().cpu()).all()):
        return False, "non-finite messages in candidate"
    low = cl.dtype in (torch.float32, torch.float16, torch.bfloat16)
    atol = _ATOL_F32 if low else _ATOL_F64
    rp, rr = _posterior(reference)
    cp, cr = _posterior(candidate)
    ep = float((rp - cp).abs().max()) if rp.numel() else 0.0
    er = float((rr - cr).abs().max()) if rr.numel() else 0.0
    ok = ep <= atol and er <= atol
    return ok, (
        f"max|dpost|={ep:.2e} max|drho|={er:.2e} (atol {atol:g}); sweeps "
        f"ref={reference['sweeps']} cand={candidate['sweeps']}; messages={rl.numel()}"
    )
