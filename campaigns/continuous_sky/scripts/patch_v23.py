"""Build smacs_four_v23.py from v22: linear multiresolution cubic B-spline sky field for the zip model (4 Nov 2026).
  * class BSplineField: drop-in for ZipField (forward(tx, ty) -> (N, K), .size), linear in its coefficients
  * --zip-field {mlp,bspline} (default mlp: unchanged), --bspline-spacings "8,4,2,1,0.5" (knot spacings, target px)
  * LOG["zip_field"] = {"kind", "spacings", "params"}
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


BSPLINE = r'''class BSplineField(nn.Module):
    """Linear multiresolution field a_k(x, y) = scale * sum_l sum_ij c_l[k, i, j] B3((x - x_i)/h_l) B3((y - y_j)/h_l),
    B3 the uniform cubic B-spline (support 4 knots, C2). No MLP, no nonlinearity; coefficients start at zero.
    Same interface as ZipField: forward(tx, ty) -> (N, K), attribute size; coordinates in target pixels (0-based
    centres), the field spans [-0.5, size-0.5].
    Level l has knot spacing h_l (target px). With u = (x + 0.5)/h_l the knots sit at u = i, i = -2 .. ceil(size/h_l)+2
    (a margin of 2 knots on every side of the field, n_l = ceil(size/h_l) + 5 coefficients per axis). Index i is the x
    (column) axis, j the y (row) axis: c_l has shape (K, n_l, n_l) = (K, i, j).
    Outside the field: u is clamped to [-1, ceil(size/h_l)+1) per axis, i.e. positions further out than the margin
    knots read the value at the clamp (constant extension, zero gradient w.r.t. position, the margin coefficients are
    used). Inside the field the levels form a partition of unity (sum of weights = 1). Evaluation gathers the 4x4
    support of each point, in chunks of `chunk` points (activation-checkpointed when gradients are needed)."""

    def __init__(self, size, k, spacings=(8, 4, 2, 1, 0.5), scale=1.0, chunk=262144):
        super().__init__()
        self.size, self.k, self.scale, self.chunk = size, k, scale, chunk
        self.spacings = tuple(float(h) for h in spacings)
        self.ncs = [int(math.ceil(size / h - 1e-9)) + 5 for h in self.spacings]
        self.coefs = nn.ParameterList([nn.Parameter(torch.zeros(k, n, n)) for n in self.ncs])

    @staticmethod
    def _taps(u, n):
        """u (N,) knot units -> flat index offsets/weights along one axis: idx (N, 4) in [0, n), w (N, 4)."""
        u = u.clamp(-1.0, float(n - 4))  # n - 4 = ceil(size/h) + 1, the last margin knot but one
        f = torch.floor(u).clamp(-1.0, float(n - 5))  # taps f-1..f+2 -> storage 0..n-1; at u = n-4: f = n-5, t = 1
        t = u - f
        t2, t3 = t * t, t * t * t
        w = torch.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], -1)
        idx = (f.long() + 2)[:, None] + torch.arange(-1, 3, device=u.device)[None]  # knot i -> storage i + 2
        return idx, w

    def _eval(self, tx, ty, *coefs):
        out = 0.0
        for c, h, n in zip(coefs, self.spacings, self.ncs):
            ix, wx = self._taps((tx + 0.5) / h, n)
            iy, wy = self._taps((ty + 0.5) / h, n)
            cf = c.reshape(self.k, -1)
            for a in range(4):
                for b in range(4):
                    out = out + cf[:, ix[:, a] * n + iy[:, b]] * (wx[:, a] * wy[:, b])[None]
        return out  # (K, N)

    def forward(self, tx, ty):  # (N,), (N,) -> (N, K)
        coefs = tuple(self.coefs)
        N = tx.shape[0]
        if N <= self.chunk:
            return self._eval(tx, ty, *coefs).T * self.scale
        need = torch.is_grad_enabled() and any(c.requires_grad for c in coefs)
        parts = []
        for i in range(0, N, self.chunk):
            a, b = tx[i:i + self.chunk], ty[i:i + self.chunk]
            if need:
                parts.append(torch.utils.checkpoint.checkpoint(self._eval, a, b, *coefs, use_reentrant=False))
            else:
                parts.append(self._eval(a, b, *coefs))
        return torch.cat(parts, 1).T * self.scale


class ZipSkyModel(nn.Module):
    """m_i(p)'''
rep("import torch.nn.functional as F\n", "import torch.nn.functional as F\nimport torch.utils.checkpoint\n")
rep('''class ZipSkyModel(nn.Module):
    """m_i(p)''', BSPLINE)

rep('''    ap.add_argument("--zip-hidden", type=int, default=96)
''', '''    ap.add_argument("--zip-hidden", type=int, default=96)
    ap.add_argument("--zip-field", choices=("mlp", "bspline"), default="mlp", help="mlp: multiresolution feature grids + MLP (ZipField); bspline: linear multiresolution cubic B-splines (BSplineField)")
    ap.add_argument("--bspline-spacings", default="8,4,2,1,0.5", help="knot spacings in target pixels, coarse to fine, for --zip-field bspline")
''')

rep('''        zfield = ZipField(a.size, a.spec_k, levels=tuple(int(v) for v in a.zip_levels.split(",")), feat=a.zip_feat,
                          hidden=a.zip_hidden, scale=float(sig_b.mean()) * 20).to(dev)
''', '''        if a.zip_field == "bspline":
            zfield = BSplineField(a.size, a.spec_k, spacings=tuple(float(v) for v in a.bspline_spacings.split(",")),
                                  scale=float(sig_b.mean()) * 20).to(dev)
        else:
            zfield = ZipField(a.size, a.spec_k, levels=tuple(int(v) for v in a.zip_levels.split(",")), feat=a.zip_feat,
                              hidden=a.zip_hidden, scale=float(sig_b.mean()) * 20).to(dev)
        n_field = sum(p_.numel() for p_ in zfield.parameters())
        LOG["zip_field"] = {"kind": a.zip_field, "spacings": (a.bspline_spacings if a.zip_field == "bspline" else a.zip_levels),
                            "params": n_field}
        tick(f"zip field {a.zip_field}: {n_field} parameters")
''')
open(dst, "w").write(s)
print("wrote", dst)
