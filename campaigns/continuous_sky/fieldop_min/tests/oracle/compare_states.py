"""Compare two saved model states tensor by tensor (rule A of PRUNE_2026-11-19.md).

    python tests/oracle/compare_states.py REFERENCE.pt NEW.pt

Each file is a dictionary name -> tensor, as written by `tests/synth_small.py --save-state` (torch.save of model.state_dict(), CPU tensors).
For every name the report says "identical" (same dtype, same shape, same bytes), "different" (shared name, any difference: the largest absolute and relative
difference is given) or "only in reference" / "only in new". A tensor that exists in one state only is not a difference: the pruned package no longer has the
tensors of the terms that were deleted. The exit status is 1 if any SHARED tensor differs (or a shared name has another dtype or shape), else 0.
Bit-identical means the raw bytes are equal, so a NaN compares equal to the same NaN and -0.0 differs from 0.0.
"""
from __future__ import annotations

import sys

import torch


def load(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(st, dict):
        raise SystemExit(f"{path}: not a state dictionary")
    return {k: v for k, v in st.items() if torch.is_tensor(v)}


def same_bytes(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if not a.numel():
        return True
    a, b = a.contiguous().reshape(-1), b.contiguous().reshape(-1)
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def compare(ref, new):
    """-> (rows, n_different); rows = [(name, status, detail)] in the order of the reference state, then the names only in the new one."""
    rows, n_diff = [], 0
    for k in ref:
        if k not in new:
            rows.append((k, "only in reference", ""))
            continue
        a, b = ref[k], new[k]
        if same_bytes(a, b):
            rows.append((k, "identical", f"{tuple(a.shape)} {str(a.dtype).replace('torch.', '')}"))
            continue
        n_diff += 1
        if a.dtype != b.dtype or a.shape != b.shape:
            rows.append((k, "different", f"dtype or shape: {tuple(a.shape)} {a.dtype} against {tuple(b.shape)} {b.dtype}"))
            continue
        d = (a.double() - b.double()).abs()
        rel = d.max() / a.double().abs().max().clamp(min=1e-300)
        rows.append((k, "different", f"max |diff| {float(d.max()):.3e}, relative to max |reference| {float(rel):.3e}, {int((d > 0).sum())} of {a.numel()} elements"))
    rows += [(k, "only in new", "") for k in new if k not in ref]
    return rows, n_diff


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    ref, new = load(argv[1]), load(argv[2])
    rows, n_diff = compare(ref, new)
    for k, s, d in rows:
        print(f"{s:18s} {k}  {d}")
    n_id = sum(1 for r in rows if r[1] == "identical")
    n_ref = sum(1 for r in rows if r[1] == "only in reference")
    n_new = sum(1 for r in rows if r[1] == "only in new")
    print(f"\nsummary: {n_id} identical, {n_diff} different, {n_ref} only in reference, {n_new} only in new, of {len(ref)} reference and {len(new)} new tensors")
    print("RESULT: " + ("FAIL (a shared tensor differs)" if n_diff else "PASS (every shared tensor is bit-identical)"))
    return 1 if n_diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
