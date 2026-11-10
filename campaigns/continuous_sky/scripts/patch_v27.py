"""v26 -> v27: --zip-warm-solver cg. For a field linear in its parameters (the B-spline field), the warm start
min ||G c - A0||^2 (G: coefficients -> spectral coefficients at the grid nodes) is solved by preconditioned conjugate gradients (level-wise Jacobi, 1/h_l^2) on the
normal equations G^T G c = G^T A0, with G^T applied by autograd (vector-Jacobian product). Exact for a linear field;
not valid for the MLP field (refused)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)


rep("""            optw = torch.optim.Adam(m.sky.parameters(), lr=a.zip_warm_lr)
            for it_w in range(20 if a.quick else a.zip_warm_steps):
                optw.zero_grad()
                lw = ((m.grid_coeffs() - A0) / sig_b.mean()).pow(2).mean()
                lw.backward()
                optw.step()
""", """            if a.zip_warm_solver == "cg":
                assert a.zip_field == "bspline", "CG warm start needs a field linear in its parameters"
                lw = cg_warm_start(m, A0, float(sig_b.mean()), 20 if a.quick else a.zip_warm_steps)
            else:
                optw = torch.optim.Adam(m.sky.parameters(), lr=a.zip_warm_lr)
                for it_w in range(20 if a.quick else a.zip_warm_steps):
                    optw.zero_grad()
                    lw = ((m.grid_coeffs() - A0) / sig_b.mean()).pow(2).mean()
                    lw.backward()
                    optw.step()
""")
rep("""def zip_capacity(a, init,""", '''def cg_warm_start(m, A0, sig, iters):
    """v27: conjugate gradients on G^T G c = G^T A0 for a linear field (G c = m.grid_coeffs()). Returns mse/sig^2."""
    params = [p_ for p_ in m.sky.parameters() if p_.requires_grad]

    def G(c):
        with torch.no_grad():
            for p_, v in zip(params, c):
                p_.copy_(v)
        return m.grid_coeffs()

    def GT(r):  # vector-Jacobian product at any point (linear field)
        out = m.grid_coeffs()
        g = torch.autograd.grad(out, params, grad_outputs=r)
        return [x.detach() for x in g]

    def dot(u, v):
        return sum((x * y).sum() for x, y in zip(u, v))

    # Jacobi-like preconditioner: the Gram diagonal of a level grows as h_l^2 (each coefficient covers h_l^2 nodes)
    hs = getattr(m.sky, "spacings", None) or [1.0] * len(params)
    Minv = [1.0 / max(h, 1.0) ** 2 for h in hs]
    x = [torch.zeros_like(p_) for p_ in params]
    b = GT(A0)
    r = [bi.clone() for bi in b]
    z = [mi * ri for mi, ri in zip(Minv, r)]
    p = [zi.clone() for zi in z]
    rs = dot(r, z)
    for k in range(iters):
        Gp = G(p)
        Ap = GT(Gp)
        alpha = rs / dot(p, Ap).clamp(min=1e-30)
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        z = [mi * ri for mi, ri in zip(Minv, r)]
        rs_new = dot(r, z)
        if k % 100 == 0 or k == iters - 1:
            with torch.no_grad():
                mse = float(((G(x) - A0) / sig).pow(2).mean())
            tick(f"CG warm start iteration {k}: mse/sig2 {mse:.4f}")
        if rs_new.abs().sqrt() < 1e-9 * b[0].abs().max():
            break
        p = [zi + (rs_new / rs) * pi for zi, pi in zip(z, p)]
        rs = rs_new
    with torch.no_grad():
        mse = (G(x) - A0) / sig
    return mse.pow(2).mean()


def zip_capacity(a, init,''')
rep('''    ap.add_argument("--zip-warm-lr", type=float, default=5e-3)\n''',
    '''    ap.add_argument("--zip-warm-lr", type=float, default=5e-3)
    ap.add_argument("--zip-warm-solver", choices=("adam", "cg"), default="adam", help="v27: cg = conjugate gradients (linear fields only)")\n''')
open(dst, "w").write(s)
print("wrote", dst)
