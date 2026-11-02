"""Finite-volume solver for the compressible Euler equations.

Planar (x, y) or axisymmetric (x, r) geometry on a uniform grid, with
- HLLC approximate Riemann fluxes (Toro, "Riemann Solvers and Numerical
  Methods for Fluid Dynamics", ch. 10),
- second-order MUSCL reconstruction of primitive variables with a minmod
  limiter, and
- second-order strong-stability-preserving Runge-Kutta (Heun) time stepping.

Axisymmetric form, with u along the axis x and v radial:

    d(U)/dt + dF/dx + (1/r) d(r G)/dr = [0, 0, p/r, 0]

so radial face fluxes are weighted by the face radius and the pressure
term p/r appears in the radial momentum equation. Written this way a
uniform gas at rest stays exactly at rest, which a test checks.

State arrays have shape (4, nx, nr): rho, rho*u, rho*v, E.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GAMMA = 1.4
NG = 2  # ghost cells per side


def prim_to_cons(rho, u, v, p, g=GAMMA):
    return np.stack([rho, rho * u, rho * v, p / (g - 1) + 0.5 * rho * (u**2 + v**2)])


def cons_to_prim(U, g=GAMMA):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (g - 1) * (U[3] - 0.5 * rho * (u**2 + v**2))
    return rho, u, v, p


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def hll_flux(WL, WR, normal: int, g=GAMMA):
    """HLL flux (two-wave). More dissipative than HLLC at contacts and shear
    layers, but free of the carbuncle instability at strong grid-aligned shocks."""
    rl, ul, vl, pl = WL
    rr, ur, vr, pr = WR
    if normal == 0:
        nl, tl, nr_, tr = ul, vl, ur, vr
    else:
        nl, tl, nr_, tr = vl, ul, vr, ur
    El = pl / (g - 1) + 0.5 * rl * (ul**2 + vl**2)
    Er = pr / (g - 1) + 0.5 * rr * (ur**2 + vr**2)
    cl = np.sqrt(g * pl / rl)
    cr = np.sqrt(g * pr / rr)
    SL = np.minimum(nl - cl, nr_ - cr)
    SR = np.maximum(nl + cl, nr_ + cr)
    FL = np.stack([rl * nl, rl * nl * nl + pl, rl * nl * tl, (El + pl) * nl])
    FR = np.stack([rr * nr_, rr * nr_ * nr_ + pr, rr * nr_ * tr, (Er + pr) * nr_])
    UL = np.stack([rl, rl * nl, rl * tl, El])
    UR = np.stack([rr, rr * nr_, rr * tr, Er])
    Fhll = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
    F = np.where(SL >= 0, FL, np.where(SR <= 0, FR, Fhll))
    if normal == 1:
        F = F[[0, 2, 1, 3]]
    return F


def hllc_flux(WL, WR, normal: int, g=GAMMA):
    """HLLC flux across faces with normal along x (normal=0) or r (normal=1).

    WL, WR: primitive states (rho, u, v, p) stacked on axis 0, each (4, ...).
    Returns the flux of (rho, rho u, rho v, E) in the normal direction.
    """
    rl, ul, vl, pl = WL
    rr, ur, vr, pr = WR
    # Normal (n) and tangential (t) velocity components.
    if normal == 0:
        nl, tl, nr_, tr = ul, vl, ur, vr
    else:
        nl, tl, nr_, tr = vl, ul, vr, ur
    El = pl / (g - 1) + 0.5 * rl * (ul**2 + vl**2)
    Er = pr / (g - 1) + 0.5 * rr * (ur**2 + vr**2)
    cl = np.sqrt(g * pl / rl)
    cr = np.sqrt(g * pr / rr)

    # Davis wave-speed estimates.
    SL = np.minimum(nl - cl, nr_ - cr)
    SR = np.maximum(nl + cl, nr_ + cr)
    SM = (pr - pl + rl * nl * (SL - nl) - rr * nr_ * (SR - nr_)) / (rl * (SL - nl) - rr * (SR - nr_))

    def phys(r, n, t, p, E):
        return np.stack([r * n, r * n * n + p, r * n * t, (E + p) * n])

    def star(r, n, t, p, E, S):
        fac = r * (S - n) / (S - SM)
        return np.stack([fac, fac * SM, fac * t, fac * (E / r + (SM - n) * (SM + p / (r * (S - n))))])

    FL = phys(rl, nl, tl, pl, El)
    FR = phys(rr, nr_, tr, pr, Er)
    UL = np.stack([rl, rl * nl, rl * tl, El])
    UR = np.stack([rr, rr * nr_, rr * tr, Er])
    FsL = FL + SL * (star(rl, nl, tl, pl, El, SL) - UL)
    FsR = FR + SR * (star(rr, nr_, tr, pr, Er, SR) - UR)

    F = np.where(SL >= 0, FL, np.where(SM >= 0, FsL, np.where(SR > 0, FsR, FR)))
    if normal == 1:  # swap momentum components back to (x, r) order
        F = F[[0, 2, 1, 3]]
    return F


@dataclass
class Grid:
    nx: int
    nr: int
    lx: float
    lr: float
    axisymmetric: bool = True
    dx: float = field(init=False)
    dr: float = field(init=False)

    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dr = self.lr / self.nr

    @property
    def xc(self):
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def rc(self):
        return (np.arange(self.nr) + 0.5) * self.dr


def _reconstruct(W, axis):
    """MUSCL face states on interior faces along `axis` (1 = x, 2 = r).

    W has ghost cells. Returns (WL, WR) at the faces between consecutive
    interior-adjacent cells: faces 0..n where n is the number of interior cells.
    """
    d = np.diff(W, axis=axis)
    sl = [slice(None)] * 3
    sr = [slice(None)] * 3
    sl[axis] = slice(0, -1)
    sr[axis] = slice(1, None)
    slope = _minmod(d[tuple(sl)], d[tuple(sr)])  # one per cell 1..N-2 (padded indices)
    # cells with a slope: padded index 1..N-2; faces we need: between padded cells
    # NG-1 | NG ... NG+n-1 | NG+n.
    c = [slice(None)] * 3
    left_cells = [slice(None)] * 3
    right_cells = [slice(None)] * 3
    n_pad = W.shape[axis]
    left_cells[axis] = slice(NG - 1, n_pad - NG)       # cell left of each face
    right_cells[axis] = slice(NG, n_pad - NG + 1)      # cell right of each face
    sl_left = [slice(None)] * 3
    sl_right = [slice(None)] * 3
    sl_left[axis] = slice(NG - 2, n_pad - NG - 1)      # slope index = padded index - 1
    sl_right[axis] = slice(NG - 1, n_pad - NG)
    WL = W[tuple(left_cells)] + 0.5 * slope[tuple(sl_left)]
    WR = W[tuple(right_cells)] - 0.5 * slope[tuple(sl_right)]
    del c
    return WL, WR


class JetSolver:
    """Underexpanded jet from a sonic nozzle of diameter D = 1 at x = 0, r < 0.5.

    Units: ambient density and pressure are 1. The jet has stagnation
    pressure `p0` and stagnation temperature equal to the ambient
    temperature (cold-flow test conditions). Boundaries:
      x = 0, r < 0.5   : fixed sonic exit state (inflow)
      x = 0, r >= 0.5  : solid wall (nozzle face plate)
      r = 0            : symmetry axis
      x = Lx, r = Lr   : zero-gradient outflow
    """

    def __init__(self, p0_over_pa: float, grid: Grid, cfl: float = 0.4, g: float = GAMMA,
                 fix_ambient_pressure: bool = True, flux: str = "hll"):
        self.fix_ambient_pressure = fix_ambient_pressure
        self.flux = {"hllc": hllc_flux, "hll": hll_flux}[flux]
        self.grid = grid
        self.g = g
        self.cfl = cfl
        self.p0 = p0_over_pa
        rho_a, p_a = 1.0, 1.0
        # Sonic exit conditions from isentropic relations (M = 1).
        t_ratio = 2.0 / (g + 1)
        self.p_exit = self.p0 * t_ratio ** (g / (g - 1))
        rho0 = rho_a * self.p0 / p_a  # T0 = T_ambient
        self.rho_exit = rho0 * t_ratio ** (1 / (g - 1))
        self.u_exit = np.sqrt(g * self.p_exit / self.rho_exit)
        self.ambient = (rho_a, p_a)

        nx, nr = grid.nx, grid.nr
        self.U = prim_to_cons(np.full((nx, nr), rho_a), np.zeros((nx, nr)), np.zeros((nx, nr)), np.full((nx, nr), p_a), g)
        self.t = 0.0
        self.steps = 0
        self.jet_rows = grid.rc < 0.5

    # ----- boundary conditions -------------------------------------------------
    def _pad(self, W):
        rho, u, v, p = W
        P = np.pad(np.stack([rho, u, v, p]), ((0, 0), (NG, NG), (NG, NG)), mode="edge")
        # Symmetry axis (or planar wall) at r = 0: mirror, flip radial velocity.
        P[:, :, :NG] = P[:, :, 2 * NG - 1:NG - 1:-1] if NG > 1 else P[:, :, NG:NG + 1]
        P[2, :, :NG] *= -1
        # x = 0 face: wall everywhere, then overwrite the nozzle exit.
        P[:, :NG, :] = P[:, 2 * NG - 1:NG - 1:-1, :]
        P[1, :NG, :] *= -1
        jet = np.zeros(P.shape[2], dtype=bool)
        jet[NG:NG + self.grid.nr] = self.jet_rows
        jet[:NG] = True  # ghost cells below the axis belong to the jet core
        P[0, :NG, jet] = self.rho_exit
        P[1, :NG, jet] = self.u_exit
        P[2, :NG, jet] = 0.0
        P[3, :NG, jet] = self.p_exit
        if self.fix_ambient_pressure:
            # Far-field boundaries (x = Lx, r = Lr): where the normal flow is
            # subsonic, impose the ambient pressure and extrapolate the rest.
            # Pure zero-gradient boundaries let the domain pressure drift.
            p_a = self.ambient[1]
            for sl, vel in (((slice(None), slice(-NG, None), slice(None)), 1),
                            ((slice(None), slice(None), slice(-NG, None)), 2)):
                block = P[sl]
                c = np.sqrt(self.g * block[3] / block[0])
                sub = np.abs(block[vel]) < c
                block[3] = np.where(sub, p_a, block[3])
                P[sl] = block
        return P

    # ----- spatial operator ---------------------------------------------------
    def rhs(self, U):
        g, grid = self.g, self.grid
        W = cons_to_prim(U, g)
        P = self._pad(W)
        dUdt = np.zeros_like(U)

        # x-direction fluxes: interior r rows only.
        Px = P[:, :, NG:-NG]
        WL, WR = _reconstruct(Px, axis=1)
        Fx = self.flux(WL, WR, normal=0, g=g)
        dUdt -= (Fx[:, 1:, :] - Fx[:, :-1, :]) / grid.dx

        # r-direction fluxes.
        Pr = P[:, NG:-NG, :]
        WL, WR = _reconstruct(Pr, axis=2)
        Fr = self.flux(WL, WR, normal=1, g=g)
        if grid.axisymmetric:
            rf = np.arange(grid.nr + 1) * grid.dr
            rc = grid.rc
            dUdt -= (Fr[:, :, 1:] * rf[1:] - Fr[:, :, :-1] * rf[:-1]) / (rc * grid.dr)
            dUdt[2] += W[3] / rc
        else:
            dUdt -= (Fr[:, :, 1:] - Fr[:, :, :-1]) / grid.dr
        return dUdt

    def max_dt(self):
        rho, u, v, p = cons_to_prim(self.U, self.g)
        c = np.sqrt(self.g * p / rho)
        smax = max(np.max(np.abs(u) + c) / self.grid.dx, np.max(np.abs(v) + c) / self.grid.dr)
        return self.cfl / smax

    def step(self, dt=None):
        dt = self.max_dt() if dt is None else dt
        U1 = self.U + dt * self.rhs(self.U)
        self.U = 0.5 * (self.U + U1 + dt * self.rhs(U1))
        self.t += dt
        self.steps += 1
        return dt

    def run(self, t_end: float, max_steps: int = 10**7):
        while self.t < t_end and self.steps < max_steps:
            self.step(min(self.max_dt(), t_end - self.t))
        return self

    # ----- derived fields -----------------------------------------------------
    def primitives(self):
        return cons_to_prim(self.U, self.g)

    def mach(self):
        rho, u, v, p = self.primitives()
        return np.sqrt(u**2 + v**2) / np.sqrt(self.g * p / rho)
