"""v36 -> v37 (6 Nov 2026): --seed S (default None = v36 behaviour, bit-identical). With S: torch, numpy and python random seeded
at start. GPU kernels are not made deterministic, so repeated runs with the same seed may still differ slightly (stated)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = '''    a = ap.parse_args()
'''
assert s.count(old) == 1
s = s.replace(old, '''    ap.add_argument("--seed", type=int, default=None, help="v37: seed torch/numpy/random at start (GPU kernels not deterministic)")
    a = ap.parse_args()
    if a.seed is not None:
        import random as _random
        _random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
''')
open(dst, "w").write(s)
print("ok")
