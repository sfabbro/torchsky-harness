"""v27 -> v28: --zip-warm-os N. The warm start fitted the field only at the grid nodes. A linear field with more
coefficients than nodes (B-spline knots at 1 px or finer, several levels) then has free structure between nodes, which
the deterministic renderer (2x grid) sees: v27 CG warm start reached mse/sig2 1.06 at the nodes but validation S/N>5
138 after 15 min. v28 fits the warm start on the N-times oversampled grid against the bilinearly upsampled coadd
coefficients (the coadd has no information between nodes; bilinear is the stated assumption)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:60], s.count(old))
    s = s.replace(old, new)


# a grid evaluator on an oversampled grid, used by both warm-start paths
rep('''    def grid_coeffs(self):  # K H W at the target pixel centres (outputs, priors)''', '''    def grid_coeffs_os(self, os_):  # v28: K (H os) (H os) on the os-times oversampled grid
        H = self.sky.size
        u = (torch.arange(H * os_, device=self.lam_nodes.device, dtype=torch.float32) + 0.5) / os_ - 0.5
        yy, xx = torch.meshgrid(u, u, indexing="ij")
        return self.sky(xx.flatten(), yy.flatten()).T.reshape(-1, H * os_, H * os_)

    def grid_coeffs(self):  # K H W at the target pixel centres (outputs, priors)''')
rep('''            A0 = spectral_init(init, lam_nodes, w_nodes, zbasis)
''', '''            A0 = spectral_init(init, lam_nodes, w_nodes, zbasis)
            if a.zip_warm_os > 1:  # v28: target on the oversampled grid (bilinear between coadd nodes)
                A0 = F.interpolate(A0[None], scale_factor=a.zip_warm_os, mode="bilinear", align_corners=False)[0]
                m.warm_os = a.zip_warm_os
''')
rep('''                    lw = ((m.grid_coeffs() - A0) / sig_b.mean()).pow(2).mean()''',
    '''                    lw = ((m.grid_coeffs_os(a.zip_warm_os) - A0) / sig_b.mean()).pow(2).mean()''')
# cg_warm_start: use the oversampled evaluator when set
rep('''        return m.grid_coeffs()

    def GT(r):  # vector-Jacobian product at any point (linear field)
        out = m.grid_coeffs()''', '''        return m.grid_coeffs_os(getattr(m, "warm_os", 1))

    def GT(r):  # vector-Jacobian product at any point (linear field)
        out = m.grid_coeffs_os(getattr(m, "warm_os", 1))''')
rep('''    ap.add_argument("--zip-warm-lr", type=float, default=5e-3)\n''',
    '''    ap.add_argument("--zip-warm-lr", type=float, default=5e-3)
    ap.add_argument("--zip-warm-os", type=int, default=1, help="v28: warm start on an N-times oversampled grid (bilinear target)")\n''')
open(dst, "w").write(s)
print("wrote", dst)
