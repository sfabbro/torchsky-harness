"""Stage 1 of make_figs.py (kept in one place by make_figs.py importing it): cache band images and kernel profiles."""
import sys, os, time, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/claude/work/flux_check"); sys.path.insert(0, "/home/claude/work/kernel_check")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"); os.makedirs(CACHE, exist_ok=True)
import flux_ratio as FR
STATES = {"grid": "grid004", "items": "items004b"}
def band_images(name):
    s = torch.load(f"/home/claude/work/states/{name}_state.pt", map_location="cpu", weights_only=False)
    A, pitch = FR.sky_coeffs(s)
    return np.stack([torch.einsum("k,khw->hw", s["Phi_band"][f].to(A.dtype), A).numpy() for f in range(6)]), pitch
if __name__ == "__main__":
    for key, name in STATES.items():
        t = time.time(); im, p = band_images(name); np.savez(f"{CACHE}/band_{key}.npz", im=im, pitch=p); print(name, im.shape, p, time.time() - t, im.min(), im.max())
