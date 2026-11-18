"""The tests written before the defaults of 18 Nov 2026 (DEFAULTS_V2_2026-11-18.md) test mechanisms, not default values: they keep running with the old
defaults (legacy_defaults, one reference throughput instrument -1 and the whole-support kernel normalisation), unless a test passes its own values.
tests/test_defaults_v2.py tests the new defaults and passes legacy_defaults=False explicitly."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fieldop.cli as _cli  # noqa: E402

_orig_default_config = _cli.default_config


def default_config(**kw):
    kw.setdefault("legacy_defaults", True)
    kw.setdefault("gauge_ref_inst", -1)
    kw.setdefault("kernel_norm_radius", 0.0)
    return _orig_default_config(**kw)


_cli.default_config = default_config
