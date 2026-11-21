"""Backend-matrix adapter: ``k_resample`` with the bilinear kernel (``F.grid_sample``).

Same geometry, inputs, timed stage and tolerance as ``k_resample.py`` (Lanczos-3);
only the interpolation kernel differs.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def _load_base():
    try:
        from . import k_resample as base  # type: ignore[no-redef]

        return base
    except ImportError:
        path = Path(__file__).with_name("k_resample.py")
        spec = importlib.util.spec_from_file_location(
            "_backend_matrix_k_resample", path
        )
        assert spec is not None and spec.loader is not None
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        return base


_base = _load_base()

KERNEL = "bilinear"
CALL_STYLE = _base.CALL_STYLE
SIZES = dict(_base.SIZES)
DTYPES = dict(_base.DTYPES)
NOTES = _base._notes(KERNEL)
check = _base.make_check(KERNEL)
items = _base.items
variants: Callable[[torch.device], dict[str, Callable[[dict], Any]]] = _base.variants


def make_inputs(size_key: str, device: torch.device, dtype: torch.dtype) -> dict:
    return _base.make_inputs_for(KERNEL, size_key, device, dtype)
