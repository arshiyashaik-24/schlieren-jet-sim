"""Flow-solver validation: exact Riemann problems, conservation, quiescent gas."""
import numpy as np
import pytest

from jetlab.euler import NG, Grid, JetSolver, cons_to_prim, prim_to_cons
from jetlab.riemann import sample


class ShockTube(JetSolver):
    """Planar solver with zero-gradient x boundaries and a Riemann initial state."""

    def __init__(self, left, right, nx=400, direction="x", flux="hll"):
        self.direction = direction
        n_long, n_short = nx, 4
        grid = Grid(nx=n_long, nr=n_short, lx=1.0, lr=n_short / n_long, axisymmetric=False)
        if direction == "r":
            grid = Grid(nx=n_short, nr=n_long, lx=n_short / n_long, lr=1.0, axisymmetric=False)
        super().__init__(p0_over_pa=2.0, grid=grid, flux=flux)
        coord = grid.xc if direction == "x" else grid.rc
        side = coord < 0.5
        shape = (grid.nx, grid.nr)
        mask = side[:, None] if direction == "x" else side[None, :]
        mask = np.broadcast_to(mask, shape)
        rho = np.where(mask, left[0], right[0])
        vel = np.where(mask, left[1], right[1])
        p = np.where(mask, left[2], right[2])
        u, v = (vel, np.zeros(shape)) if direction == "x" else (np.zeros(shape), vel)
        self.U = prim_to_cons(rho, u, v, p)

    def _pad(self, W):
        return np.pad(np.stack(W), ((0, 0), (NG, NG), (NG, NG)), mode="edge")


SOD = ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
STRONG = ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01))  # Toro test 3: very strong shock


def l1_error(case, nx, t_end, direction="x", flux="hll"):
    left, right = case
    s = ShockTube(left, right, nx=nx, direction=direction, flux=flux).run(t_end)
    rho, u, v, p = cons_to_prim(s.U)
    if direction == "x":
        x, rho_num = s.grid.xc, rho[:, 0]
    else:
        x, rho_num = s.grid.rc, rho[0, :]
    rho_ex = sample(left, right, (x - 0.5) / t_end)[0]
    return np.mean(np.abs(rho_num - rho_ex)), s


@pytest.mark.parametrize("flux", ["hll", "hllc"])
def test_sod_matches_exact_solution(flux):
    err, s = l1_error(SOD, 400, 0.2, flux=flux)
    assert err < 5e-3


def test_sod_converges_with_grid_refinement():
    e1, _ = l1_error(SOD, 100, 0.2)
    e2, _ = l1_error(SOD, 400, 0.2)
    # Shocks limit any scheme to ~first order in L1; require a clear decrease.
    assert e2 < 0.4 * e1


@pytest.mark.parametrize("flux", ["hll", "hllc"])
def test_strong_shock_is_stable_and_accurate(flux):
    err, s = l1_error(STRONG, 400, 0.012, flux=flux)
    rho, u, v, p = cons_to_prim(s.U)
    assert np.all(rho > 0) and np.all(p > 0)
    assert err < 0.15  # density jump of ~6 across a Mach ~200 shock


def test_radial_direction_matches_axial_direction():
    ex, _ = l1_error(SOD, 200, 0.2, direction="x")
    er, _ = l1_error(SOD, 200, 0.2, direction="r")
    assert er == pytest.approx(ex, rel=1e-9)


def test_mass_and_energy_conserved_in_closed_tube():
    s = ShockTube(*SOD, nx=200)
    m0, e0 = s.U[0].sum(), s.U[3].sum()
    s.run(0.1)  # waves have not reached the boundaries yet
    assert s.U[0].sum() == pytest.approx(m0, rel=1e-12)
    assert s.U[3].sum() == pytest.approx(e0, rel=1e-12)


def test_axisymmetric_gas_at_rest_stays_at_rest():
    # A well-balanced axisymmetric scheme must not generate flow from p/r.
    grid = Grid(nx=24, nr=24, lx=2.0, lr=2.0, axisymmetric=True)
    s = JetSolver(p0_over_pa=1.0, grid=grid)
    s.jet_rows[:] = False  # no nozzle: closed wall at x = 0
    for _ in range(50):
        s.step()
    rho, u, v, p = s.primitives()
    assert np.max(np.abs(u)) < 1e-12 and np.max(np.abs(v)) < 1e-12


def test_hllc_resolves_contact_more_sharply_than_hll():
    # The reason HLLC exists: it keeps the contact wave; HLL smears it.
    e_hll, _ = l1_error(SOD, 200, 0.2, flux="hll")
    e_hllc, _ = l1_error(SOD, 200, 0.2, flux="hllc")
    assert e_hllc < e_hll
