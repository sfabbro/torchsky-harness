"""v25 -> v26: --zip-warm-lr (warm-start Adam learning rate; v20-v25 hard-coded 5e-3). The linear B-spline field starts at
zero with coefficients of order 50 (scale 20 sigma); at 5e-3 its warm start ended at mse/sig2 1865 after 3000 steps
(MLP field: 0.02). A CPU test (96^2, compact sources up to 1000 sigma, 1500 steps) gives mse/sig2 16.8 / 0.36 / 0.011 at
lr 5e-3 / 5e-2 / 0.2."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = "            optw = torch.optim.Adam(m.sky.parameters(), lr=5e-3)\n"
assert s.count(old) == 1
s = s.replace(old, "            optw = torch.optim.Adam(m.sky.parameters(), lr=a.zip_warm_lr)\n")
old = '    ap.add_argument("--zip-warm-steps", type=int, default=800)\n'
assert s.count(old) == 1
s = s.replace(old, old + '    ap.add_argument("--zip-warm-lr", type=float, default=5e-3)\n')
open(dst, "w").write(s)
print("wrote", dst)
