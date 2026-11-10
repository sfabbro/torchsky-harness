import sys, types, torch, tempfile, pathlib
sys.path.insert(0, "/home/claude/work/scripts")
import smacs_four_v21 as S
torch.manual_seed(0)
H, K = 64, 8
lam = torch.tensor([[0.85, 0.9, 0.95], [1.4, 1.5, 1.6], [1.9, 2.0, 2.1], [2.6, 2.8, 3.0], [3.3, 3.55, 3.8], [4.1, 4.4, 4.7]])
w = torch.full((6, 3), 1 / 3)
init = torch.rand(6, H, H) + 5 * (torch.rand(6, H, H) > 0.98)
sig = torch.full((6,), 0.1)
src = init > 1.0
out = pathlib.Path(tempfile.mkdtemp())
st = {"sky.coeffs.s": torch.rand(K, H, H), "psf.x": torch.zeros(3)}
torch.save(st, out / "t.pt")
a = types.SimpleNamespace(spec_k=K, init="drizzle", capacity_target_state=str(out / "t.pt"), size=H,
    capacity_configs="base:4:96:8,16,32,64;big:8:128:8,16,32,64,128", capacity_min=0.05, capacity_lr=5e-3)
r = S.zip_capacity(a, init, sig, src, lam, w, "cpu", out)
for k, v in r.items(): print(k, v["params"], v["steps"], round(v["coeff_mse_sig2"], 3), round(v["band_mse_sig2_all"], 3), round(v["band_mse_sig2_bright"], 3))
print(sorted(p.name for p in out.iterdir()))
