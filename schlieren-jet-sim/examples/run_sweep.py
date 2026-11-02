"""Run underexpanded-jet simulations and store results in data/.

    python examples/run_sweep.py sweep        # pressure-ratio sweep at 32 cells/D
    python examples/run_sweep.py convergence  # p0/pa = 10 at 16, 24, 32, 48 cells/D

Each case runs past the start-up transient, then records the Mach disk
position every 0.5 time units (units: D / ambient speed of sound) to give a
mean and a standard deviation, and saves the final flow field for imaging.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from jetlab.analysis import crist_correlation, mach_disk_from_flow
from jetlab.euler import Grid, JetSolver

DATA = Path(__file__).resolve().parents[1] / "data"
T_SETTLE, T_END, DT_SAMPLE = 16.0, 30.0, 0.5


def domain(ratio):
    xm = crist_correlation(ratio)
    lx = float(max(6.0, np.ceil(1.6 * xm + 2.5)))
    lr = float(max(3.0, np.ceil(1.1 * xm)))
    return lx, lr


def run_case(args):
    ratio, cells_per_d, tag = args
    lx, lr = domain(ratio)
    grid = Grid(nx=int(lx * cells_per_d), nr=int(lr * cells_per_d), lx=lx, lr=lr)
    s = JetSolver(ratio, grid)
    t0 = time.time()
    s.run(T_SETTLE)
    disks, rho_axis, mach_axis = [], [], []
    t = T_SETTLE
    while t < T_END - 1e-9:
        t += DT_SAMPLE
        s.run(t)
        rho, u, v, p = s.primitives()
        m_axis = s.mach()[:, 0]
        rho_axis.append(rho[:, 0])
        mach_axis.append(m_axis)
        disks.append(mach_disk_from_flow(rho[:, 0], m_axis, grid.xc))
    rho, u, v, p = s.primitives()
    out = DATA / f"{tag}_ratio{ratio:g}_n{cells_per_d}.npz"
    np.savez_compressed(out, ratio=ratio, cells_per_d=cells_per_d, lx=lx, lr=lr,
                        x=grid.xc, r=grid.rc, rho=rho, u=u, v=v, p=p, mach=s.mach(),
                        disks=np.array(disks), rho_axis=np.array(rho_axis), mach_axis=np.array(mach_axis), wall_s=time.time() - t0, steps=s.steps)
    q1, med, q3 = np.nanpercentile(disks, [25, 50, 75])
    print(f"{tag} p0/pa={ratio:g} n={cells_per_d}: disk median {med:.3f} [IQR {q1:.3f}-{q3:.3f}] "
          f"(Crist {crist_correlation(ratio):.3f}), {s.steps} steps, {time.time() - t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        cases = [(r, 32, "sweep") for r in (6, 10, 15, 20, 30)]
    else:
        cases = [(10, n, "conv") for n in (16, 24, 32, 48)]
    cases.sort(key=lambda c: -c[0] * c[1])  # longest first for better load balance
    with Pool(2) as pool:
        for _ in pool.imap_unordered(run_case, cases):
            pass
