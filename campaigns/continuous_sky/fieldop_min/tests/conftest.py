"""Test defaults. The tests test mechanisms, not default values: they keep running with one reference throughput instrument (--gauge-ref-inst -1) and the
whole-support kernel normalisation (--kernel-norm-radius 0) unless a test passes its own values (the package defaults are -1 with one instrument and 2 arcsec)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fieldop.cli as _cli  # noqa: E402

_orig_default_config = _cli.default_config


def default_config(**kw):
    kw.setdefault("gauge_ref_inst", -1)
    kw.setdefault("kernel_norm_radius", 0.0)
    return _orig_default_config(**kw)


_cli.default_config = default_config
