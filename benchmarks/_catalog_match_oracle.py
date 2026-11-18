"""Independent crossmatch oracles and adversarial sky geometry (NumPy only).

Nothing here imports torchsky. Separations use the atan2 form on float64 unit
vectors. A pair whose separation lies within ``tie_band`` of its threshold is
float64-ambiguous: it is counted and reported, but excluded from set
comparisons. ``mpmath_separation`` gives an extended-precision reference for
boundary samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi
TIE_BAND_ARCSEC = 1e-8


def unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def separations(la, lb, ra, rb, *, chunk: int = 1024) -> np.ndarray:
    """Dense ``[N, M]`` great-circle separations in arcsec."""
    a, b = unit(la, lb), unit(ra, rb)
    out = np.empty((a.shape[0], b.shape[0]))
    for s in range(0, a.shape[0], chunk):
        aa = a[s : s + chunk, None, :]
        cross = np.linalg.norm(np.cross(aa, b[None, :, :]), axis=-1)
        dot = np.clip(np.einsum("ijk,ljk->ij", aa, b[None]), -1.0, 1.0)
        out[s : s + chunk] = np.arctan2(cross, dot) * ARCSEC_PER_RAD
    return out


def mpmath_separation(ra1, dec1, ra2, dec2, *, dps: int = 50) -> float:
    """Great-circle separation (arcsec) in ``dps``-digit arithmetic."""
    import mpmath as mp

    with mp.workdps(dps):
        r1, d1, r2, d2 = (mp.radians(mp.mpf(float(v))) for v in (ra1, dec1, ra2, dec2))
        v1 = (mp.cos(d1) * mp.cos(r1), mp.cos(d1) * mp.sin(r1), mp.sin(d1))
        v2 = (mp.cos(d2) * mp.cos(r2), mp.cos(d2) * mp.sin(r2), mp.sin(d2))
        cx = v1[1] * v2[2] - v1[2] * v2[1]
        cy = v1[2] * v2[0] - v1[0] * v2[2]
        cz = v1[0] * v2[1] - v1[1] * v2[0]
        dot = sum(x * y for x, y in zip(v1, v2))
        return float(mp.atan2(mp.sqrt(cx * cx + cy * cy + cz * cz), dot) * 180 * 3600 / mp.pi)


def threshold(n: int, m: int, rl, rr, combine: str) -> np.ndarray:
    rl = np.broadcast_to(np.asarray(rl, dtype=np.float64), (n,))[:, None]
    if rr is None:
        return np.broadcast_to(rl, (n, m))
    rr = np.broadcast_to(np.asarray(rr, dtype=np.float64), (m,))[None, :]
    if combine == "sum":
        return rl + rr
    if combine == "quadrature":
        return np.sqrt(rl * rl + rr * rr)
    if combine == "max":
        return np.maximum(rl, rr)
    raise ValueError(combine)


def band_mask(sep: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.abs(sep - thr) <= TIE_BAND_ARCSEC + 1e-12 * thr


@dataclass
class SetComparison:
    missing: int
    extra: int
    ambiguous: int
    expected: int

    @property
    def ok(self) -> bool:
        return self.missing == 0 and self.extra == 0


def compare_all(pairs: set[tuple[int, int]], sep: np.ndarray, thr: np.ndarray) -> SetComparison:
    inside = sep <= thr
    band = band_mask(sep, thr)
    sure = set(zip(*np.nonzero(inside & ~band)))
    amb = set(zip(*np.nonzero(band)))
    sure = {(int(i), int(j)) for i, j in sure}
    amb = {(int(i), int(j)) for i, j in amb}
    return SetComparison(
        missing=len(sure - pairs), extra=len(pairs - sure - amb), ambiguous=len(amb), expected=len(sure)
    )


def compare_best(best: dict[int, int], sep: np.ndarray, thr: np.ndarray) -> SetComparison:
    """Nearest right row per left row within the threshold; ties to lowest index."""
    masked = np.where(sep <= thr, sep, np.inf)
    band = band_mask(sep, thr).any(axis=1)
    expected: dict[int, int] = {}
    ambiguous: set[int] = set()
    if masked.shape[1]:
        arg = np.argmin(masked, axis=1)  # first minimum: lowest index on exact ties
        val = masked[np.arange(masked.shape[0]), arg]
        for i in range(masked.shape[0]):
            if band[i]:
                ambiguous.add(i)
                continue
            if not np.isfinite(val[i]):
                continue
            near = np.abs(masked[i] - val[i]) <= TIE_BAND_ARCSEC
            near[arg[i]] = False
            if near.any() and (masked[i][near] != val[i]).any():
                ambiguous.add(i)
                continue
            expected[i] = int(arg[i])
    missing = sum(1 for i, j in expected.items() if best.get(i) != j)
    extra = sum(1 for i in best if i not in expected and i not in ambiguous)
    return SetComparison(missing=missing, extra=extra, ambiguous=len(ambiguous), expected=len(expected))


def nearest(sep: np.ndarray, k: int, cap: float = math.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[N, k]`` neighbour indices (-1 beyond the cap), separations, and ambiguity mask."""
    order = np.argsort(sep, axis=1, kind="stable")
    srt = np.take_along_axis(sep, order, axis=1)
    take = min(k, sep.shape[1])
    idx = np.full((sep.shape[0], k), -1, dtype=np.int64)
    out = np.full((sep.shape[0], k), np.inf)
    inside = srt[:, :take] <= cap
    idx[:, :take] = np.where(inside, order[:, :take], -1)
    out[:, :take] = np.where(inside, srt[:, :take], np.inf)
    # Any near-tie (not exact) among the first k+1 neighbours makes the order
    # float64-ambiguous; exact ties are resolved by the lowest index.
    amb = np.zeros(sep.shape[0], dtype=bool)
    upto = min(take + 1, sep.shape[1])
    if upto > 1:
        gap = np.diff(srt[:, :upto], axis=1)
        amb |= ((gap <= TIE_BAND_ARCSEC) & (gap > 0.0)).any(axis=1)
    if math.isfinite(cap):
        amb |= (np.abs(srt[:, :take] - cap) <= TIE_BAND_ARCSEC).any(axis=1)
    return idx, out, amb


# ------------------------------------------------------------------ geometry


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def uniform_sky(n: int, seed: int):
    g = _rng(seed)
    return g.uniform(0.0, 360.0, n), np.degrees(np.arcsin(g.uniform(-1.0, 1.0, n)))


def jitter(ra, dec, sigma_arcsec: float, seed: int):
    """Random tangent-plane offsets, pole safe."""
    g = _rng(seed)
    v = unit(ra, dec)
    east = np.stack([-np.sin(np.radians(ra)), np.cos(np.radians(ra)), np.zeros_like(ra)], -1)
    north = np.cross(v, east)
    pole = np.abs(v[:, 2]) > 1.0 - 1e-15
    east[pole] = (1.0, 0.0, 0.0)
    north[pole] = (0.0, 1.0, 0.0)
    s = math.radians(sigma_arcsec / 3600.0)
    w = v + g.normal(0, s, len(ra))[:, None] * east + g.normal(0, s, len(ra))[:, None] * north
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    return np.degrees(np.arctan2(w[:, 1], w[:, 0])) % 360.0, np.degrees(np.arcsin(np.clip(w[:, 2], -1, 1)))


def pole_field(n: int, seed: int, south: bool = False):
    g = _rng(seed)
    ra = g.uniform(0, 360, n)
    dec = 90.0 - g.uniform(0, 0.02, n)
    dec[: max(1, n // 50)] = 90.0
    return (ra, -dec) if south else (ra, dec)


def wrap_field(n: int, seed: int, half_width: float = 0.01):
    g = _rng(seed)
    return g.uniform(-half_width, half_width, n) % 360.0, g.uniform(-half_width, half_width, n)


def antimeridian_pole_field(n: int, seed: int):
    """Points straddling RA 0/360 within 0.05 deg of the north pole."""
    g = _rng(seed)
    return g.uniform(-30, 30, n) % 360.0, 90.0 - g.uniform(0, 0.05, n)


def plummer_cluster(n: int, seed: int, scale_arcsec: float = 10.0):
    g = _rng(seed)
    u = g.uniform(1e-6, 1.0, n)
    r = scale_arcsec / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    r = np.minimum(r, 50 * scale_arcsec) / 3600.0
    t = g.uniform(0, 2 * math.pi, n)
    dec = 2.0 + r * np.sin(t)
    return (150.0 + r * np.cos(t) / math.cos(math.radians(2.0))) % 360.0, dec


@dataclass
class Case:
    name: str
    left: tuple[np.ndarray, np.ndarray]
    right: tuple[np.ndarray, np.ndarray]
    radius_left: np.ndarray | float
    radius_right: np.ndarray | None = None
    combine: str = "sum"


def _duplicates(seed: int) -> Case:
    la, lb = uniform_sky(400, seed)
    return Case("coincident_duplicates", (la, lb), (np.tile(la, 3), np.tile(lb, 3)), 1.0)


def _equidistant(seed: int) -> Case:
    # Right rows on a ring of exactly equal separation around each left row.
    la, lb = wrap_field(100, seed, 0.5)
    angles = np.arange(6) * math.pi / 3.0
    off = 2.0 / 3600.0
    ra = (la[:, None] + off * np.cos(angles)[None, :] / np.cos(np.radians(lb))[:, None]).ravel() % 360.0
    rb = (lb[:, None] + off * np.sin(angles)[None, :]).ravel()
    return Case("equidistant_ties", (la, lb), (ra, rb), 3.0)


def _radius_ladder(seed: int) -> list[Case]:
    out = []
    for r in (1e-3, 1e-2, 0.1, 1.0, 10.0, 60.0, 3600.0):
        la, lb = uniform_sky(600, seed) if r >= 60.0 else wrap_field(600, seed, 0.02)
        ra, rb = jitter(la, lb, max(r, 1e-3), seed + 1)
        out.append(Case(f"radius_ladder_{r:g}arcsec", (la, lb), (ra, rb), r))
    return out


def adversarial_cases(seed: int = 7, scale: int = 1) -> list[Case]:
    n = 800 * scale
    cases: list[Case] = []
    la, lb = pole_field(n, seed)
    cases.append(Case("pole_north", (la, lb), jitter(la, lb, 5.0, seed + 1), 30.0))
    la, lb = pole_field(n, seed + 2, south=True)
    cases.append(Case("pole_south", (la, lb), jitter(la, lb, 5.0, seed + 3), 30.0))
    la, lb = wrap_field(n, seed + 4)
    cases.append(Case("ra_wrap", (la, lb), jitter(la, lb, 2.0, seed + 5), 5.0))
    la, lb = antimeridian_pole_field(n, seed + 6)
    cases.append(Case("antimeridian_at_pole", (la, lb), jitter(la, lb, 3.0, seed + 7), 10.0))
    cases.append(_duplicates(seed + 8))
    cases.append(_equidistant(seed + 9))
    cases.extend(_radius_ladder(seed + 10))
    la, lb = uniform_sky(n, seed + 20)
    ra, rb = jitter(la, lb, 30.0, seed + 21)
    g = _rng(seed + 22)
    cases.append(Case("per_row_radius_5dex", (la, lb), (ra, rb), 1.8e-3 * 10 ** g.uniform(0, 5, n)))
    cases.append(Case("plummer_cluster", plummer_cluster(2 * n, seed + 23), plummer_cluster(2 * n, seed + 24), 0.5))
    la, lb = plummer_cluster(n, seed + 25)
    ra, rb = plummer_cluster(n, seed + 26)
    rl, rr = g.uniform(0, 0.5, n), g.uniform(0, 0.5, n)
    rr[:3] = 10.0
    for combine in ("sum", "quadrature", "max"):
        cases.append(Case(f"two_sided_{combine}", (la, lb), (ra, rb), rl, rr, combine))
    return cases


def time_call(fn: Callable[[], object], runs: int) -> float:
    import time

    fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))
