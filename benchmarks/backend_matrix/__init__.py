"""Per-kernel backend matrix: eager torch vs torch.compile vs native, per device.

Kernel modules are ``k_*.py`` files in this directory; ``run_matrix.py`` loads
them by file path, so the directory can be copied to a machine without the
harness installed and run as ``python run_matrix.py ...``.
"""
