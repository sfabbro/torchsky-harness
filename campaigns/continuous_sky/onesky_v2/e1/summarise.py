"""Read every results/*.json and print a Markdown table; save figs/e1_by_band.png."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = ['1-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-1024']


def load():
    rows = []
    for f in glob.glob(os.path.join(HERE, 'results', '*.json')):
        with open(f) as fh: r = json.load(fh)
        if all(k in r for k in ('method', 'budget', 'chi2_truth', 'frac_err_by_band')): rows.append(r)
        else: print(f'<!-- skipped {os.path.basename(f)}: not a single-result record -->')
    rows.sort(key=lambda r: (r['budget'], r['chi2_truth']))
    return rows


def label(r):
    v = r['variant'].replace('seed0_zoom1.0', '').strip('_')
    return r['method'] + (f' ({v})' if v else '')


def table(rows):
    head = ['method', 'target', 'budget', 'fitted numbers (image / shared)', 'chi2_truth', 'psnr (dB)', 'fit s'] + [f'err {b} px' for b in BANDS]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        fe = r['frac_err_by_band']
        out.append('| ' + ' | '.join([label(r), r['target'], str(r['budget']), f"{r['n_params_image']} / {r['n_params_shared']}",
                                       f"{r['chi2_truth']:.3f}", f"{r['psnr']:.1f}", f"{r['fit_seconds']:.0f}"] + [f"{fe[b]:.3f}" for b in BANDS]) + ' |')
    return '\n'.join(out)


def figure(rows):
    rows = [r for r in rows if r['target'] == 'noisy' and r['n'] == 256 and r['variant'] == 'seed0_zoom1.0']
    budgets = sorted({r['budget'] for r in rows}); methods = sorted({r['method'] for r in rows})
    cmap = plt.get_cmap('tab10'); col = {m: cmap(i % 10) for i, m in enumerate(methods)}
    mk = dict(zip(methods, 'osD^vP*Xp<'))
    fig, ax = plt.subplots(1, max(1, len(budgets)), figsize=(5.6 * max(1, len(budgets)), 5), sharey=True, squeeze=False)
    for a, b in zip(ax[0], budgets):
        for r in rows:
            if r['budget'] != b: continue
            y = [r['frac_err_by_band'][k] for k in BANDS]
            a.plot(range(len(BANDS)), y, marker=mk[r['method']], color=col[r['method']], label=r['method'], lw=1.6, ms=5)
        a.set_yscale('log'); a.set_xticks(range(len(BANDS))); a.set_xticklabels(BANDS, rotation=40, ha='right')
        a.set_xlabel('scale band (wavelength, pixels)'); a.set_title(f'budget {b}'); a.grid(alpha=0.3, which='both')
        a.axhline(1.0, color='grey', ls=':', lw=1)
    ax[0][0].set_ylabel('fractional error (RMS error / RMS truth in band)')
    h, l = ax[0][-1].get_legend_handles_labels()
    if h: ax[0][-1].legend(fontsize=9, loc='best')
    fig.suptitle('E1: error against truth by scale band (fitted to the noisy image)')
    fig.tight_layout(); os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'figs', 'e1_by_band.png'), dpi=120)


if __name__ == '__main__':
    rows = load()
    print(table(rows)); figure(rows)
