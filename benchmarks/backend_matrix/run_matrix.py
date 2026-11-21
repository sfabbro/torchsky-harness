#!/usr/bin/env python
"""Backend matrix driver: time each kernel as eager torch, torch.compile and native.

Every ``k_*.py`` module in the module directory (default: this file's
directory) provides::

    SIZES = {"small": ..., "medium": ..., "large": ...}
    DTYPES = {"cpu": torch.float64, "cuda": torch.float64, "mps": torch.float32}
    NOTES = "..."                                   # optional
    def make_inputs(size_key, device, dtype) -> dict  # seeded, on device
    def variants(device) -> dict[str, callable]       # "eager", "compile", "native" (cpu), ...
    def items(size_key) -> int
    def check(reference, candidate) -> tuple[bool, str]

A variant is called as ``fn(**inputs)``. If its signature cannot bind the input
keys but accepts one positional argument it is called as ``fn(inputs)``; a
module may force either with ``CALL_STYLE = "kwargs" | "dict"``.

Modules are imported by file path (as submodules of a synthetic package, so
relative imports of sibling helpers work), so the directory can be shipped to
a remote job without the harness installed::

    python run_matrix.py --device cpu --device cuda --out cuda.jsonl
    python run_matrix.py --summarize cpu_mps.jsonl cuda.jsonl

Timing protocol per (kernel, size, device, variant): ``variants(device)`` is
called and inputs are built once per (kernel, size, device) group, after
``torch._dynamo.reset()``, and again before each further ``compile*`` variant
of the group, since wrappers of one code object share Dynamo's cache
(``--dynamo-reset module`` resets only per kernel; never inside timing, never
before eager/native variants). The first call is timed alone as the
cold time (it includes torch.compile compilation); the warm calls are each
bracketed by device synchronization. The cold output is passed to the module's
``check(reference, candidate)`` exactly as returned (on its device, in its
dtype; the driver never converts outputs because a one-step device+dtype
``.to()`` is wrong on MPS in torch 2.14.1). The reference is ``native`` on CPU
if present, else ``eager`` on CPU, in the module's CPU dtype, on the same seeded
inputs.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import inspect
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DRIVER_VERSION = "1"
HERE = Path(__file__).resolve().parent
SIZE_ORDER = ("small", "medium", "large")
DEVICE_ORDER = ("cpu", "mps", "cuda")
PKG_NAME = "_backend_matrix_kernels"
ERR_MAX = 400
REQUIRED_ATTRS = ("SIZES", "DTYPES", "make_inputs", "variants", "items", "check")

torch: Any = None  # imported in main() after cache environment variables are set


class CallTimeout(Exception):
    """Raised from SIGALRM when a single call exceeds the wall timeout."""


# --------------------------------------------------------------------------- helpers


def _err(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".strip()
    return msg if len(msg) <= ERR_MAX else msg[: ERR_MAX - 3] + "..."


def _is_timeout(exc: BaseException | None) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, CallTimeout):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


@contextmanager
def _deadline(seconds: float | None):
    """Best-effort wall timeout: SIGALRM fires between Python bytecodes only."""
    usable = (
        seconds
        and seconds > 0
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield
        return

    def _handler(signum, frame):
        raise CallTimeout(f"call exceeded {seconds:g} s wall timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def _empty_cache(dev: str) -> None:
    try:
        if dev == "cuda":
            torch.cuda.empty_cache()
        elif dev == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _make_caller(module: Any, fn: Callable, inputs: dict) -> Callable[[], Any]:
    style = getattr(module, "CALL_STYLE", None)
    if style == "kwargs":
        return lambda: fn(**inputs)
    if style == "dict":
        return lambda: fn(inputs)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return lambda: fn(**inputs)
    try:
        sig.bind(**inputs)
        return lambda: fn(**inputs)
    except TypeError:
        pass
    try:
        sig.bind(inputs)
        return lambda: fn(inputs)
    except TypeError:
        return lambda: fn(**inputs)


def _timed(
    call: Callable[[], Any], dev: str, timeout: float | None
) -> tuple[float, Any]:
    with _deadline(timeout):
        _sync(dev)
        t0 = time.perf_counter()
        out = call()
        _sync(dev)
        t1 = time.perf_counter()
    return t1 - t0, out


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank quantile of an ascending list."""
    idx = max(0, math.ceil(q * len(sorted_vals)) - 1)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _dynamo_counter(group: str, key: str | None = None) -> int:
    try:
        from torch._dynamo.utils import counters

        return int(counters[group][key] if key else sum(counters[group].values()))
    except Exception:
        return 0


def _dynamo_reset() -> None:
    try:
        import torch._dynamo
        from torch._dynamo.utils import counters

        torch._dynamo.reset()
        counters.clear()
    except Exception:
        pass


# --------------------------------------------------------------------------- modules


def discover(module_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(module_dir.glob("k_*.py"))}


def select_modules(available: dict[str, Path], spec: str | None) -> list[str]:
    if not spec:
        return list(available)
    chosen: list[str] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        pats = [tok] if tok.startswith("k_") else [tok, "k_" + tok]
        hits = [n for n in available if any(fnmatch.fnmatchcase(n, p) for p in pats)]
        if not hits:
            raise SystemExit(
                f"--kernels: nothing matches {tok!r}; available: {', '.join(available) or '(none)'}"
            )
        chosen.extend(h for h in hits if h not in chosen)
    return chosen


def load_module(module_dir: Path, stem: str) -> types.ModuleType:
    pkg = sys.modules.get(PKG_NAME)
    if pkg is None or list(getattr(pkg, "__path__", [])) != [str(module_dir)]:
        pkg = types.ModuleType(PKG_NAME)
        pkg.__path__ = [str(module_dir)]
        sys.modules[PKG_NAME] = pkg
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    name = f"{PKG_NAME}.{stem}"
    spec = importlib.util.spec_from_file_location(name, module_dir / f"{stem}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    missing = [a for a in REQUIRED_ATTRS if not hasattr(mod, a)]
    if missing:
        raise AttributeError(f"{stem}.py lacks {', '.join(missing)}")
    return mod


def _short(stem: str) -> str:
    return stem[2:] if stem.startswith("k_") else stem


def _variant_order(names: list[str], ref_name: str | None, dev: str) -> list[str]:
    """CPU reference variant first (it seeds the reference), then eager, compile*, native, others."""

    def key(n: str) -> int:
        if dev == "cpu" and n == ref_name:
            return 0
        return {"eager": 1, "native": 3}.get(n, 2 if n.startswith("compile") else 4)

    return sorted(names, key=lambda n: (key(n), names.index(n)))


def _skipped(patterns: list[str], dev: str, variant: str) -> bool:
    for p in patterns:
        if "/" in p:
            pd, pv = p.split("/", 1)
            if fnmatch.fnmatchcase(dev, pd) and fnmatch.fnmatchcase(variant, pv):
                return True
        elif fnmatch.fnmatchcase(variant, p):
            return True
    return False


# --------------------------------------------------------------------------- environment


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if out:
                return out
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _git_info(path: Path) -> dict[str, Any]:
    try:
        top = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if top.returncode != 0:
            return {}
        sha = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"torchsky_git_sha": sha, "torchsky_git_dirty": bool(dirty)}
    except Exception:
        return {}


def environment_record(
    args: argparse.Namespace, devices: list[str], unavailable: list[str]
) -> dict:
    rec: dict[str, Any] = {
        "kind": "env",
        "driver_version": DRIVER_VERSION,
        "stamp": args.stamp,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "os_release": platform.release(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cpu_name": _cpu_name(),
        "devices": devices,
        "unavailable_devices": unavailable,
        "compile_cache": args.compile_cache,
        "dynamo_reset": args.dynamo_reset,
        "sizes": args.sizes,
        "reps": args.reps or "adaptive",
        "target_s": args.target_seconds,
        "timeout_s": args.timeout,
        "guard_s": args.guard_seconds,
        "skip_variants": args.skip_variants,
        "module_dir": Path(args.module_dir).name,
    }
    device_names: dict[str, str] = {}
    for dev in devices:
        if dev == "cuda":
            try:
                device_names[dev] = torch.cuda.get_device_name()
            except Exception as exc:
                device_names[dev] = _err(exc)
        else:  # cpu, and mps (Apple GPU shares the SoC name)
            device_names[dev] = rec["cpu_name"]
    rec["device_names"] = device_names
    try:
        import numpy

        rec["numpy"] = numpy.__version__
    except Exception:
        rec["numpy"] = None
    try:
        import torchsky

        rec["torchsky"] = getattr(torchsky, "__version__", None)
        rec["torchsky_native"] = (
            importlib.util.find_spec("torchsky._cpp_ext") is not None
        )
        rec.update(_git_info(Path(torchsky.__file__).resolve().parent))
    except Exception as exc:
        rec["torchsky"] = None
        rec["torchsky_error"] = _err(exc)
    return rec


# --------------------------------------------------------------------------- reference


class Reference:
    """CPU reference output for one (module, size), computed at most once."""

    def __init__(self, module: Any, size: str, timeout: float | None):
        self.module = module
        self.size = size
        self.timeout = timeout
        self.output: Any = None
        self.label: str | None = None
        self.error: str | None = None
        self.failed: set[str] = set()
        try:
            names = list(module.variants(torch.device("cpu")))
        except Exception as exc:
            names = []
            self.error = f"variants(cpu): {_err(exc)}"
        self.candidates = [n for n in ("native", "eager") if n in names]
        if not self.candidates and self.error is None:
            self.error = "module has neither cpu 'native' nor 'eager'"

    @property
    def ready(self) -> bool:
        return self.label is not None

    @property
    def wanted(self) -> str | None:
        return next((n for n in self.candidates if n not in self.failed), None)

    def adopt(self, name: str, output_cpu: Any) -> None:
        self.output, self.label, self.error = output_cpu, f"cpu/{name}", None

    def ensure(self) -> None:
        while not self.ready and self.wanted is not None:
            name = self.wanted
            try:
                cpu = torch.device("cpu")
                torch.manual_seed(0)
                inputs = self.module.make_inputs(
                    self.size, cpu, self.module.DTYPES["cpu"]
                )
                fn = self.module.variants(cpu)[name]
                _, out = _timed(
                    _make_caller(self.module, fn, inputs), "cpu", self.timeout
                )
                self.adopt(name, out)
            except Exception as exc:
                self.failed.add(name)
                self.error = f"reference cpu/{name}: {_err(exc)}"

    def compare(self, candidate: Any) -> tuple[bool | None, str]:
        """check(reference as computed on cpu, candidate as returned on its device)."""
        self.ensure()
        if not self.ready:
            return None, f"no reference ({self.error})"
        try:
            ok, msg = self.module.check(self.output, candidate)
            return bool(ok), str(msg)[:ERR_MAX]
        except Exception as exc:
            return False, f"check raised {_err(exc)}"


# --------------------------------------------------------------------------- timing


def run_variant(
    module: Any,
    fn: Callable,
    inputs: dict,
    dev: str,
    name: str,
    ref: Reference,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"status": "ok"}
    call = _make_caller(module, fn, inputs)
    mem_base = 0
    mps_peak = 0
    if dev == "cuda":
        _empty_cache(dev)
        torch.cuda.reset_peak_memory_stats()
        mem_base = torch.cuda.memory_allocated()

    def note_mem() -> None:
        nonlocal mps_peak
        if dev == "mps":
            try:
                mps_peak = max(mps_peak, int(torch.mps.driver_allocated_memory()))
            except Exception:
                pass

    # Cold call: compilation (if any) + first execution. Dynamo resets happen in
    # the caller, never here; counters are deltas. stats.unique_graphs counts
    # Dynamo captures on every path (frames.ok is not bumped under fullgraph=True).
    gb_before = _dynamo_counter("graph_break")
    graphs_before = _dynamo_counter("stats", "unique_graphs")
    fx_hits_before = _dynamo_counter("inductor", "fxgraph_cache_hit")
    try:
        cold, out = _timed(call, dev, args.timeout)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        _empty_cache(dev)
        if dev == "cpu" and not ref.ready and name == ref.wanted:
            ref.failed.add(name)  # do not retry it as a standalone reference
            ref.error = f"reference cpu/{name}: {_err(exc)}"
        return {
            "status": "timeout" if _is_timeout(exc) else "error",
            "error": _err(exc),
            "stage": "cold",
        }
    rec["cold_s"] = cold
    note_mem()
    if "compile" in name:
        rec["graph_breaks"] = _dynamo_counter("graph_break") - gb_before
        rec["graphs_compiled"] = (
            _dynamo_counter("stats", "unique_graphs") - graphs_before
        )
        rec["fxgraph_cache_hits"] = (
            _dynamo_counter("inductor", "fxgraph_cache_hit") - fx_hits_before
        )
    # Correctness on the cold output. The candidate goes to check() exactly as
    # returned (device and dtype untouched): modules choose device-specific
    # tolerances and do their own transfer (one-step device+dtype .to() is wrong
    # on MPS in torch 2.14.1, so the driver never converts outputs).
    if dev == "cpu" and not ref.ready and name == ref.wanted:
        ref.adopt(name, out)
        rec["correct"], rec["check_msg"] = True, "reference"
    else:
        rec["correct"], rec["check_msg"] = ref.compare(out)
    rec["reference"] = ref.label
    del out

    # Warm calls.
    graphs_before = _dynamo_counter("stats", "unique_graphs")
    samples: list[float] = []
    n_target = args.reps or args.max_reps
    try:
        while len(samples) < n_target:
            t, out = _timed(call, dev, args.timeout)
            note_mem()
            del out
            samples.append(t)
            if not args.reps and len(samples) == 1:
                n_target = int(
                    min(
                        args.max_reps,
                        max(
                            args.min_reps, math.ceil(args.target_seconds / max(t, 1e-9))
                        ),
                    )
                )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        rec.update(
            status="timeout" if _is_timeout(exc) else "error",
            error=_err(exc),
            stage="warm",
        )
        _empty_cache(dev)
    if "graphs_compiled" in rec:
        rec["warm_recompiles"] = (
            _dynamo_counter("stats", "unique_graphs") - graphs_before
        )
    if samples:
        s = sorted(samples)
        med = (
            _quantile(s, 0.5)
            if len(s) % 2
            else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
        )
        rec.update(reps=len(s), min_s=s[0], median_s=med, p90_s=_quantile(s, 0.9))
    if dev == "cuda":
        peak = int(torch.cuda.max_memory_allocated())
        rec.update(
            peak_bytes=peak,
            peak_over_inputs_bytes=peak - mem_base,
            mem_kind="cuda_max_memory_allocated",
        )
    elif dev == "mps" and mps_peak:
        rec.update(peak_bytes=mps_peak, mem_kind="mps_driver_allocated_max_sampled")
    return rec


def compiler_warmup(devices: list[str], timeout: float | None) -> dict[str, Any]:
    """Compile a trivial unrelated graph once per device.

    The first torch.compile in a process pays one-time inductor/toolchain start-up
    (seconds on CPU); without this, that cost would land on whichever kernel runs
    first. The warm-up time is recorded in the env record.
    """
    out: dict[str, Any] = {}
    for dev in devices:
        try:
            _dynamo_reset()
            fn = torch.compile(lambda t: torch.cos(t) * 2.0 + 1.0)
            x = torch.arange(7, dtype=torch.float32, device=dev)
            out[dev], _ = _timed(lambda: fn(x), dev, timeout)
        except Exception as exc:
            out[dev] = _err(exc)
    _dynamo_reset()
    return out


def _fmt_s(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 1e3:.3g} ms" if v < 1 else f"{v:.3g} s"


def run(args: argparse.Namespace) -> Path:
    module_dir = Path(args.module_dir).resolve()
    available = discover(module_dir)
    stems = select_modules(available, args.kernels)
    if not stems:
        raise SystemExit(f"no k_*.py modules in {module_dir}")
    sizes = [s for s in SIZE_ORDER if s in args.sizes] + [
        s for s in args.sizes if s not in SIZE_ORDER
    ]
    skip_pats = [p for p in (args.skip_variants or "").split(",") if p]

    requested = list(dict.fromkeys(args.device or ["cpu"]))
    devices, unavailable = [], []
    for dev in sorted(requested, key=lambda d: DEVICE_ORDER.index(d)):
        ok = (
            dev == "cpu"
            or (dev == "cuda" and torch.cuda.is_available())
            or (dev == "mps" and torch.backends.mps.is_available())
        )
        (devices if ok else unavailable).append(dev)
        if not ok:
            print(f"warning: device {dev} unavailable, skipped", file=sys.stderr)
    if not devices:
        raise SystemExit("no requested device is available")

    out_path = (
        Path(args.out) if args.out else Path(f"backend_matrix_{args.stamp}.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:

        def emit(rec: dict) -> None:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()

        env = environment_record(args, devices, unavailable)
        env["compiler_warmup_s"] = (
            compiler_warmup(devices, args.timeout) if args.compile_warmup else None
        )
        emit(env)
        for stem in stems:
            kernel = _short(stem)
            _dynamo_reset()
            try:
                mod = load_module(module_dir, stem)
            except Exception as exc:
                print(f"[{kernel}] load failed: {_err(exc)}", file=sys.stderr)
                emit(
                    {
                        "kind": "module",
                        "kernel": kernel,
                        "status": "error",
                        "error": _err(exc),
                    }
                )
                continue
            emit(
                {
                    "kind": "module",
                    "kernel": kernel,
                    "status": "ok",
                    "notes": str(getattr(mod, "NOTES", ""))[:2000],
                    "sizes": {k: repr(v) for k, v in mod.SIZES.items()},
                    "dtypes": {k: str(v) for k, v in mod.DTYPES.items()},
                }
            )
            slow: dict[tuple[str, str], str] = {}
            for size in sizes:
                if size not in mod.SIZES:
                    continue
                try:
                    n_items = int(mod.items(size))
                except Exception as exc:
                    n_items = None
                    print(
                        f"[{kernel}] items({size}) failed: {_err(exc)}", file=sys.stderr
                    )
                ref = Reference(mod, size, args.timeout)
                for dev in devices:
                    base = {
                        "kind": "result",
                        "kernel": kernel,
                        "size": size,
                        "device": dev,
                        "items": n_items,
                    }
                    dtype = mod.DTYPES.get(dev)
                    if dtype is None:
                        emit(
                            {
                                **base,
                                "variant": "(inputs)",
                                "status": "error",
                                "error": f"DTYPES has no {dev!r}",
                            }
                        )
                        continue
                    base["dtype"] = str(dtype).replace("torch.", "")
                    device = torch.device(dev)
                    if args.dynamo_reset == "group":
                        # Fresh Dynamo state per (module, size, device): without it the
                        # second size of a code object recompiles with automatic dynamic
                        # shapes and cold times depend on run order.
                        _dynamo_reset()
                    try:
                        vs = dict(mod.variants(device))
                    except Exception as exc:
                        emit(
                            {
                                **base,
                                "variant": "(variants)",
                                "status": "error",
                                "error": _err(exc),
                            }
                        )
                        continue
                    try:
                        torch.manual_seed(0)
                        inputs = mod.make_inputs(size, device, dtype)
                        _sync(dev)
                    except Exception as exc:
                        _empty_cache(dev)
                        emit(
                            {
                                **base,
                                "variant": "(inputs)",
                                "status": "error",
                                "error": _err(exc),
                            }
                        )
                        print(
                            f"[{kernel}] {size} {dev}: make_inputs failed: {_err(exc)}",
                            file=sys.stderr,
                        )
                        continue
                    group: list[dict] = []
                    compiled_in_group = False
                    for name in _variant_order(list(vs), ref.wanted, dev):
                        rec = {**base, "variant": name}
                        if _skipped(skip_pats, dev, name):
                            continue
                        why = slow.get((dev, name))
                        if size == "large" and why:
                            rec.update(status="skipped", error=why)
                            if dev == "cpu" and not ref.ready and name == ref.wanted:
                                ref.failed.add(
                                    name
                                )  # too slow to recompute as the reference
                                ref.error = f"reference cpu/{name} skipped: {why}"
                        else:
                            if "compile" in name:
                                if compiled_in_group and args.dynamo_reset == "group":
                                    # Two torch.compile wrappers of one code object share
                                    # Dynamo's cache: without this the second one's cold
                                    # call is a cache hit running the first one's code.
                                    _dynamo_reset()
                                    rec["dynamo_reset_before"] = True
                                compiled_in_group = True
                            rec.update(
                                run_variant(mod, vs[name], inputs, dev, name, ref, args)
                            )
                            if rec.get("graphs_compiled") == 0:
                                rec["cold_cache_hit"] = True
                        if rec.get("median_s") and n_items:
                            rec["items_per_s"] = n_items / rec["median_s"]
                        if size == "medium":
                            if rec.get("status") == "timeout":
                                slow[(dev, name)] = "medium timed out"
                            elif (rec.get("median_s") or 0) > args.guard_seconds:
                                slow[(dev, name)] = (
                                    f"medium median {rec['median_s']:.1f} s > {args.guard_seconds:g} s"
                                )
                        group.append(rec)
                        flag = {True: "ok", False: "WRONG", None: "unchecked"}[
                            rec.get("correct")
                        ]
                        print(
                            f"[{kernel}] {size:<6} {dev}/{name:<18} {rec['status']:<7} "
                            f"med {_fmt_s(rec.get('median_s')):>10}  cold {_fmt_s(rec.get('cold_s')):>10}  "
                            f"reps {rec.get('reps', 0):>2}  {flag}"
                            + (f"  {rec['error'][:120]}" if rec.get("error") else ""),
                            file=sys.stderr,
                            flush=True,
                        )
                    del inputs
                    _empty_cache(dev)
                    best_ok = min(
                        (
                            r["median_s"]
                            for r in group
                            if r.get("correct") and r.get("median_s")
                        ),
                        default=None,
                    )
                    for r in group:
                        wbf = (
                            r.get("correct") is False
                            and r.get("median_s") is not None
                            and best_ok is not None
                            and r["median_s"] < best_ok
                        )
                        r["wrong_but_fast"] = bool(wbf)
                        if wbf:
                            print(
                                f"[{kernel}] WARNING wrong-but-fast: {size} {dev}/{r['variant']} "
                                f"({r.get('check_msg')})",
                                file=sys.stderr,
                            )
                        emit(r)
            sys.modules.pop(f"{PKG_NAME}.{stem}", None)
    return out_path


# --------------------------------------------------------------------------- summary


def _ms(v: float) -> str:
    ms = v * 1e3
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 10:
        return f"{ms:.1f}"
    if ms >= 1:
        return f"{ms:.2f}"
    return f"{ms:.3g}"


def _variant_rank(v: str) -> tuple[int, str]:
    if v == "eager":
        return (0, v)
    if v.startswith("compile"):
        return (1, v)
    if v == "native":
        return (2, v)
    return (3, v)


def summarize(paths: list[str]) -> str:
    runs: list[dict] = []  # each: {"tag", "env", "results", "modules"}
    for p in paths:
        cur = None
        for i, line in enumerate(Path(p).read_text().splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: {p}:{i + 1}: not JSON, ignored", file=sys.stderr)
                continue
            kind = rec.get("kind")
            if kind == "env" or cur is None:
                env = rec if kind == "env" else {}
                cur = {
                    "tag": env.get("stamp") or Path(p).stem,
                    "env": env,
                    "results": [],
                    "modules": {},
                }
                runs.append(cur)
                if kind == "env":
                    continue
            if kind == "module":
                cur["modules"][rec.get("kernel")] = rec
            elif kind == "result":
                cur["results"].append(rec)
    tags: dict[str, int] = {}
    for r in runs:  # make run tags unique
        n = tags.get(r["tag"], 0)
        tags[r["tag"]] = n + 1
        if n:
            r["tag"] = f"{r['tag']}#{n + 1}"

    lines = ["# Backend matrix summary", "", "## Runs", ""]
    lines += [
        "| run | torch | torchsky | threads | devices | compile cache | compiler warm-up |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        e = r["env"]
        sha = e.get("torchsky_git_sha")
        ts = f"{e.get('torchsky')}" + (
            f" ({sha[:8]}{'+dirty' if e.get('torchsky_git_dirty') else ''})"
            if sha
            else ""
        )
        devs = ", ".join(f"{d}: {n}" for d, n in (e.get("device_names") or {}).items())
        wu = e.get("compiler_warmup_s") or {}
        wus = (
            ", ".join(
                f"{d}: {_fmt_s(v) if isinstance(v, float) else 'ERR'}"
                for d, v in wu.items()
            )
            or "-"
        )
        lines.append(
            f"| {r['tag']} | {e.get('torch')} | {ts} | {e.get('threads')} | {devs} | {e.get('compile_cache')} | {wus} |"
        )

    # Device keys shared by several runs get a run prefix.
    dev_runs: dict[str, set[str]] = {}
    for r in runs:
        for x in r["results"]:
            dev_runs.setdefault(x["device"], set()).add(r["tag"])

    def dkey(tag: str, dev: str) -> str:
        return f"{tag}:{dev}" if len(dev_runs.get(dev, ())) > 1 else dev

    results = [(r["tag"], x) for r in runs for x in r["results"]]
    kernels = list(dict.fromkeys(x["kernel"] for _, x in results))
    modules = {k: m for r in runs for k, m in r["modules"].items()}
    errors: list[str] = []
    wrong_fast: list[str] = []
    for k, m in modules.items():
        if m.get("status") == "error":
            errors.append(f"- `{k}` (load): {m.get('error')}")

    for kernel in kernels:
        rows = [(t, x) for t, x in results if x["kernel"] == kernel]
        cols = sorted(
            {
                (dkey(t, x["device"]), x["device"], x["variant"])
                for t, x in rows
                if not x["variant"].startswith("(")
            },
            key=lambda c: (
                DEVICE_ORDER.index(c[1]) if c[1] in DEVICE_ORDER else 9,
                c[0],
                _variant_rank(c[2]),
            ),
        )
        dkeys = list(dict.fromkeys(c[0] for c in cols))
        lines += ["", f"## {kernel}", ""]
        notes = (modules.get(kernel) or {}).get("notes")
        if notes:
            lines += [f"_{' '.join(notes.split())}_", ""]
        header = (
            ["size", "items"]
            + [f"{c[0]}/{c[2]}" for c in cols]
            + ["best vs eager", "correct", "compile cold"]
        )
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        sizes = sorted(
            {x["size"] for _, x in rows},
            key=lambda s: SIZE_ORDER.index(s) if s in SIZE_ORDER else 9,
        )
        for size in sizes:
            cell: dict[tuple[str, str], dict] = {}
            n_items = None
            for t, x in rows:
                if x["size"] != size:
                    continue
                n_items = n_items or x.get("items")
                if x["variant"].startswith("("):
                    errors.append(
                        f"- `{kernel}` {size} {dkey(t, x['device'])} {x['variant']}: {x.get('error')}"
                    )
                    continue
                cell[(dkey(t, x["device"]), x["variant"])] = x
            vals = [size, f"{n_items:,}" if n_items else "-"]
            bad: list[str] = []
            unchecked = False
            for dk, _, v in cols:
                x = cell.get((dk, v))
                if x is None:
                    vals.append("-")
                    continue
                st = x.get("status")
                label = f"{dk}/{v}"
                if x.get("correct") is False:
                    bad.append(label)
                elif x.get("correct") is None and st != "skipped":
                    unchecked = True
                if x.get("wrong_but_fast"):
                    wrong_fast.append(
                        f"- `{kernel}` {size} {label}: {_ms(x['median_s'])} ms; {x.get('check_msg')}"
                    )
                if st in ("error", "timeout"):
                    errors.append(
                        f"- `{kernel}` {size} {label} [{st}/{x.get('stage', '?')}]: {x.get('error')}"
                    )
                if x.get("median_s") is not None:
                    txt = _ms(x["median_s"]) + (
                        " ✗" if x.get("correct") is False else ""
                    )
                    if st != "ok":
                        txt += f" ({st})"
                    vals.append(txt)
                else:
                    vals.append(
                        {"error": "ERR", "timeout": "T/O", "skipped": "skip"}.get(
                            st, st or "?"
                        )
                    )
            best_parts, cold_parts = [], []
            for dk in dkeys:
                here = {
                    v: cell[(d, v)] for d, _, v in cols if d == dk and (d, v) in cell
                }
                good = {
                    v: x["median_s"]
                    for v, x in here.items()
                    if x.get("median_s") and x.get("correct") is not False
                }
                if good:
                    bv = min(good, key=good.get)
                    eager = good.get("eager")
                    ratio = f" {eager / good[bv]:.2f}x" if eager else ""
                    best_parts.append(f"{dk}: {bv}{ratio}")
                for v, x in here.items():
                    if "compile" in v and x.get("cold_s") is not None:
                        gb = x.get("graph_breaks")
                        cold_parts.append(
                            f"{dk}/{v} {_fmt_s(x['cold_s'])}"
                            + (f" ({gb} gb)" if gb else "")
                            + (" (dynamo cache hit)" if x.get("cold_cache_hit") else "")
                            + (" (fx cache hit)" if x.get("fxgraph_cache_hits") else "")
                        )
            correct = "✗ " + ", ".join(bad) if bad else ("✓?" if unchecked else "✓")
            vals += [
                "; ".join(best_parts) or "-",
                correct,
                "; ".join(cold_parts) or "-",
            ]
            lines.append("| " + " | ".join(vals) + " |")
    lines += [
        "",
        "Cells: warm median ms; ✗ = failed the module check against the CPU reference; "
        "best vs eager = fastest correct variant and eager/best median ratio on that device; "
        "✓? = some variant had no reference; gb = graph breaks.",
    ]
    lines += ["", "## Wrong but fast", ""] + (wrong_fast or ["none"])
    lines += ["", "## Errors", ""] + (errors or ["none"])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--device",
        action="append",
        choices=DEVICE_ORDER,
        help="repeatable; default cpu",
    )
    p.add_argument(
        "--kernels",
        help="comma list of module names or globs (k_ prefix optional); default all",
    )
    p.add_argument(
        "--sizes",
        default="small,medium,large",
        help="comma list (default small,medium,large)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="torch.set_num_threads (native follows it); default 4",
    )
    p.add_argument(
        "--reps", type=int, default=None, help="fixed warm reps (default adaptive)"
    )
    p.add_argument(
        "--target-seconds",
        type=float,
        default=1.0,
        help="adaptive warm budget per variant (1.0)",
    )
    p.add_argument("--min-reps", type=int, default=3)
    p.add_argument("--max-reps", type=int, default=50)
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="best-effort per-call wall timeout, s (300)",
    )
    p.add_argument(
        "--guard-seconds",
        type=float,
        default=30.0,
        help="skip large if medium median exceeds this (30)",
    )
    p.add_argument(
        "--skip-variants",
        default="",
        help="comma list of variant names/globs, or device/variant",
    )
    p.add_argument(
        "--compile-cache",
        choices=("fresh", "default"),
        default="fresh",
        help="fresh: private inductor/triton cache dir so cold times include full compilation",
    )
    p.add_argument(
        "--dynamo-reset",
        choices=("group", "module"),
        default="group",
        help="group (default): torch._dynamo.reset() per (module,size,device) group and before each "
        "further compile* variant in it; module: only per module (later compile variants of the "
        "same function may hit the cache; flagged cold_cache_hit)",
    )
    p.add_argument(
        "--no-compile-warmup",
        dest="compile_warmup",
        action="store_false",
        help="skip the per-device trivial compile that absorbs one-time inductor start-up",
    )
    p.add_argument(
        "--stamp",
        default=None,
        help="label recorded in the env record (default UTC timestamp)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output JSONL (default ./backend_matrix_<stamp>.jsonl)",
    )
    p.add_argument(
        "--module-dir",
        default=str(HERE),
        help="directory holding k_*.py (default: this directory)",
    )
    p.add_argument("--list", action="store_true", help="list kernel modules and exit")
    p.add_argument(
        "--summarize",
        nargs="+",
        metavar="JSONL",
        help="print markdown summary of result files; no run",
    )
    a = p.parse_args(argv)
    a.sizes = [s.strip() for s in a.sizes.split(",") if s.strip()]
    a.stamp = a.stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if a.min_reps < 1 or a.max_reps < a.min_reps:
        p.error("need 1 <= --min-reps <= --max-reps")
    return a


def main(argv: list[str] | None = None) -> int:
    global torch
    args = parse_args(argv)
    if args.summarize:
        sys.stdout.write(summarize(args.summarize))
        return 0
    if args.list:
        for stem in discover(Path(args.module_dir).resolve()):
            print(_short(stem))
        return 0
    if args.compile_cache == "fresh":
        cache = tempfile.mkdtemp(prefix="bm_inductor_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    import torch as _torch

    torch = _torch
    torch.set_num_threads(args.threads)
    out = run(args)
    print(f"wrote {out}", file=sys.stderr)
    sys.stdout.write(summarize([str(out)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
