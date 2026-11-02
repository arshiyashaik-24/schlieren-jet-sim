"""Generate all figures from the stored simulations (run run_sweep.py first).

    python examples/make_figures.py
"""
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jetlab.analysis import ashkenas_sherman_mach, crist_correlation, mach_disk_from_flow, mach_disk_from_image
from jetlab.optics import SchlierenSetup, mean_free_path, schlieren_image

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
})

# Physical set-up for imaging (stated in the README).
D_NOZZLE = 1e-3        # m
T_AMB = 293.15         # K
SETUP = SchlierenSetup()
PRESSURES = np.logspace(0, 5, 26)  # 1 Pa .. 1 bar
SUCCESS_TOL = 0.1      # detection counts if within 0.1 D of the true disk
N_TRIALS = 40


def load(tag):
    runs = [dict(np.load(f)) for f in sorted(DATA.glob(f"{tag}_*.npz"))]
    return sorted(runs, key=lambda d: (float(d["ratio"]), int(d["cells_per_d"])))


def image_grid(run):
    r_edges = np.r_[0.0, 0.5 * (run["r"][1:] + run["r"][:-1]), run["lr"]]
    dy = run["r"][1] - run["r"][0]
    y = np.arange(-run["lr"] + dy / 2, run["lr"], dy)
    return r_edges, y


def fig_validation():
    from jetlab.euler import cons_to_prim
    from jetlab.riemann import sample
    import importlib.util
    spec = importlib.util.spec_from_file_location("ts", ROOT / "tests" / "test_solver.py")
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    s = ts.ShockTube(*ts.SOD, nx=200).run(0.2)
    rho = cons_to_prim(s.U)[0][:, 0]
    xf = np.linspace(0, 1, 2000)
    axes[0].plot(xf, sample(*ts.SOD, (xf - 0.5) / 0.2)[0], color=MUTED, lw=1.5, label="exact")
    axes[0].plot(s.grid.xc, rho, "o", ms=3, color=BLUE, label="solver, 200 cells")
    axes[0].set(title="(a) Sod shock tube, t = 0.2", xlabel="x", ylabel="Density")
    axes[0].legend(frameon=False)

    ns = np.array([50, 100, 200, 400, 800])
    errs = [ts.l1_error(ts.SOD, n, 0.2)[0] for n in ns]
    axes[1].loglog(ns, errs, "o-", color=BLUE, label="L1 density error")
    axes[1].loglog(ns, errs[0] * (ns[0] / ns), "--", color=MUTED, lw=1.2, label="first order")
    axes[1].set(title="(b) Grid convergence", xlabel="Cells", ylabel="L1 error")
    axes[1].legend(frameon=False)
    fig.savefig(FIG / "02_solver_validation.png", dpi=160)
    plt.close(fig)
    return errs


P_DISPLAY = 1e4  # Pa, ambient pressure for the example image


def adapted_setup(eps, base=SETUP):
    """Widen the knife-edge opening so the largest deflection uses 90% of the
    linear range, as an experimenter would to avoid saturation."""
    needed = base.focal_length * np.abs(eps).max() / 0.9
    return replace(base, cutoff=max(base.cutoff, needed))


def fig_jet(run):
    r_edges, y = image_grid(run)
    x = run["x"]
    _, eps = schlieren_image(run["rho"], x, r_edges, y, D_NOZZLE, P_DISPLAY, T_AMB, "N2", SETUP)
    setup = adapted_setup(eps)
    img = setup.intensity(eps)
    x_true = mach_disk_from_flow(run["rho"][:, 0], run["mach"][:, 0], x)
    x_img, _ = mach_disk_from_image(img, x, y)
    mach = run["mach"]
    full = np.concatenate([mach[:, ::-1], mach], axis=1)
    r_full = np.r_[-run["r"][::-1], run["r"]]

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.4), constrained_layout=True)
    m = axes[0].pcolormesh(x, r_full, full.T, cmap="viridis", shading="auto", vmin=0, vmax=np.percentile(mach, 99.5))
    fig.colorbar(m, ax=axes[0], label="Mach number", pad=0.01)
    g = axes[1].pcolormesh(x, y, img.T, cmap="gray", shading="auto", vmin=0, vmax=2)
    fig.colorbar(g, ax=axes[1], label="I / I0", pad=0.01)
    lim = min(float(run["lr"]), 2.0)
    for ax in axes:
        ax.axvline(x_true, color=ORANGE, lw=1.2, ls="--")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set(ylabel="y / D", xlabel="x / D", ylim=(-lim, lim), xlim=(0, float(run["lx"])))
    axes[1].axvline(x_img, color=AQUA, lw=1.2, ls=":")
    axes[0].set_title(f"(a) Simulated jet, p0/pa = {float(run['ratio']):g}\nshock on the axis at x/D = {x_true:.2f} (dashed)", loc="left")
    axes[1].set_title(f"(b) Synthetic Schlieren, N2 at {P_DISPLAY / 1e3:g} kPa, D = 1 mm\nbrightest image feature at x/D = {x_img:.2f} (dotted)", loc="left")
    fig.savefig(FIG / "03_jet_and_schlieren.png", dpi=160)
    plt.close(fig)
    return x_true, x_img


def robust(d):
    q1, med, q3 = np.nanpercentile(d, [25, 50, 75])
    return med, med - q1, q3 - med


def fig_mach_disk(sweep, conv):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    ratios = np.array([float(r["ratio"]) for r in sweep])
    stats = np.array([robust(r["disks"]) for r in sweep])
    rr = np.linspace(4, 35, 200)
    axes[0].plot(rr, crist_correlation(rr), color=MUTED, lw=1.5, label="Crist et al. (1966), experiment")
    axes[0].errorbar(ratios, stats[:, 0], yerr=stats[:, 1:].T, fmt="o", color=BLUE, ms=5, capsize=3,
                     label="simulation (median, IQR)")
    axes[0].set(title="(a) Mach disk position", xlabel="Pressure ratio p0 / pa", ylabel="x_M / D")
    axes[0].legend(frameon=False)

    n = np.array([int(r["cells_per_d"]) for r in conv])
    cst = np.array([robust(r["disks"]) for r in conv])
    axes[1].errorbar(n, cst[:, 0], yerr=cst[:, 1:].T, fmt="o-", color=BLUE, ms=5, capsize=3, label="simulation")
    axes[1].axhline(crist_correlation(10), color=MUTED, lw=1.2, ls="--", label="Crist correlation")
    axes[1].set(title="(b) Grid convergence, p0/pa = 10", xlabel="Cells per nozzle diameter", ylabel="x_M / D")
    axes[1].legend(frameon=False)

    xs = np.linspace(0.5, 4, 200)
    axes[2].plot(xs, ashkenas_sherman_mach(xs), color=MUTED, lw=1.5, label="Ashkenas & Sherman (1966)")
    for run, col in zip([r for r in sweep if float(r["ratio"]) in (10, 30)], (BLUE, ORANGE)):
        x = run["x"]
        m = np.median(run["mach_axis"], axis=0)
        upstream = x < 0.9 * np.nanmedian(run["disks"])
        axes[2].plot(x[upstream], m[upstream], color=col, label=f"simulation, p0/pa = {float(run['ratio']):g}")
    axes[2].set(title="(c) Centreline Mach number before the disk", xlabel="x / D", ylabel="Mach number")
    axes[2].legend(frameon=False)
    fig.savefig(FIG / "04_mach_disk_vs_pressure_ratio.png", dpi=160)
    plt.close(fig)
    return ratios, stats, n, cst


def detectability(run):
    """Per-pixel SNR and noise robustness of the image detector vs ambient pressure.

    At each pressure the knife edge is opened just enough to avoid saturation.
    Success means the noisy-image detection lands within SUCCESS_TOL of the
    noise-free image detection; the systematic line-of-sight offset between
    the image feature and the on-axis shock is reported separately.
    """
    r_edges, y = image_grid(run)
    x = run["x"]
    x_true = mach_disk_from_flow(run["rho"][:, 0], run["mach"][:, 0], x)
    rng = np.random.default_rng(42)
    rows = np.abs(y) <= 0.25
    res = {}
    for gas in ("N2", "H2"):
        snr, success = [], {1: [], 100: []}
        for p in PRESSURES:
            _, eps = schlieren_image(run["rho"], x, r_edges, y, D_NOZZLE, p, T_AMB, gas, SETUP)
            setup = adapted_setup(eps)
            img = setup.intensity(eps)
            x_ref, profile = mach_disk_from_image(img, x, y)
            k = int(np.argmin(np.abs(x - x_ref)))
            peak = np.max(img[:, rows].mean(axis=1)[max(k - 3, 0):k + 4]) - 1
            snr.append(peak / setup.noise_sigma(1.0))
            for frames in success:
                hits = 0
                for _ in range(N_TRIALS):
                    xd, _ = mach_disk_from_image(setup.camera(img, rng=rng, frames=frames), x, y)
                    hits += abs(xd - x_ref) < SUCCESS_TOL
                success[frames].append(hits / N_TRIALS)
        res[gas] = (np.array(snr), {k: np.array(v) for k, v in success.items()})
    upstream = x < x_true
    rho_min = run["rho"][upstream, 0].min()
    kn = {g: mean_free_path(PRESSURES, T_AMB, g) / rho_min / D_NOZZLE for g in ("N2", "H2")}
    return res, kn, rho_min


def fig_detectability(run, res, kn):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    colors = {"N2": BLUE, "H2": ORANGE}
    for gas, (snr, succ) in res.items():
        axes[0].loglog(PRESSURES, snr, color=colors[gas], label=gas)
        axes[1].semilogx(PRESSURES, succ[1], color=colors[gas], label=f"{gas}, 1 frame")
        axes[1].semilogx(PRESSURES, succ[100], color=colors[gas], ls="--", label=f"{gas}, 100 frames")
        axes[2].loglog(PRESSURES, kn[gas], color=colors[gas], label=gas)
    axes[0].axhline(5, color=MUTED, lw=1.2, ls="--")
    axes[0].text(1.3, 6, "SNR = 5", color=MUTED)
    axes[0].set(title="(a) Mach disk signal (knife edge set to avoid saturation)", xlabel="Ambient pressure [Pa]", ylabel="SNR (single frame)")
    axes[1].set(title=f"(b) Noisy detection within ±{SUCCESS_TOL} D of noise-free", xlabel="Ambient pressure [Pa]", ylabel="Success rate", ylim=(-0.03, 1.03))
    axes[2].axhline(0.1, color=MUTED, lw=1.2, ls="--")
    axes[2].text(2e3, 0.13, "Kn = 0.1", color=MUTED)
    axes[2].text(1.3, 2e-3, "continuum (Euler / Navier-Stokes)\nunreliable above the dashed line", color=MUTED, fontsize=8)
    axes[2].set(title="(c) Is the flow still a continuum?", xlabel="Ambient pressure [Pa]", ylabel="Kn upstream of Mach disk")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Low-pressure limits at p0/pa = {float(run['ratio']):g}, D = 1 mm, f2 = 1 m, knife-edge opening >= 0.5 mm", color=INK)
    fig.savefig(FIG / "05_low_pressure_detectability.png", dpi=160)
    plt.close(fig)


def threshold(p, values, level, rising=True):
    """Lowest pressure at which `values` crosses `level` (log interpolation)."""
    above = values >= level if rising else values <= level
    if not above.any():
        return np.nan
    i = int(np.argmax(above))
    if i == 0:
        return p[0]
    lp = np.interp(level, [values[i - 1], values[i]], np.log10([p[i - 1], p[i]]))
    return 10**lp


if __name__ == "__main__":
    errs = fig_validation()
    sweep, conv = load("sweep"), load("conv")
    ratios, stats, n, cst = fig_mach_disk(sweep, conv)
    ref = next(r for r in sweep if float(r["ratio"]) == 10)
    x_true, x_img = fig_jet(ref)
    res, kn, rho_min = detectability(ref)
    fig_detectability(ref, res, kn)

    print("Sod L1 errors (50..800 cells):", np.round(errs, 5))
    for r, (m, lo, hi) in zip(ratios, stats):
        print(f"p0/pa={r:>4g}: x_M/D median {m:.3f} (-{lo:.3f}/+{hi:.3f})   Crist {crist_correlation(r):.3f}   diff {100 * (m / crist_correlation(r) - 1):+.1f}%")
    for k, (m, lo, hi) in zip(n, cst):
        print(f"convergence n={k}: median {m:.3f} (-{lo:.3f}/+{hi:.3f})")
    print(f"p0/pa=10 line-of-sight offset: image feature {x_img:.3f} vs axis shock {x_true:.3f} ({x_img - x_true:+.3f} D); min axis density upstream {rho_min:.3f} rho_a")
    for gas, (snr, succ) in res.items():
        print(f"{gas}: SNR=5 at {threshold(PRESSURES, snr, 5):.0f} Pa; 90% detection at "
              f"{threshold(PRESSURES, succ[1], 0.9):.0f} Pa (1 frame), {threshold(PRESSURES, succ[100], 0.9):.0f} Pa (100 frames); "
              f"Kn=0.1 at {threshold(PRESSURES, kn[gas], 0.1, rising=False):.1f} Pa")
