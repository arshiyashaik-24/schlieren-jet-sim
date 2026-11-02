"""Synthetic Schlieren imaging of an axisymmetric density field.

Chain: density field -> refractive index (Gladstone-Dale) -> line-of-sight
integral through the axisymmetric jet (forward Abel projection) -> ray
deflection -> knife-edge intensity -> camera photon and read noise.

Physical assumptions and parameter values are collected in `SchlierenSetup`
and stated in the README; they are typical lab values, not ASML's setup.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_B = 1.380649e-23        # J/K
R_GAS = 8.314462618       # J/(mol K)
P_STP, T_STP = 101325.0, 273.15

# Refractivity (n - 1) at STP near 589 nm, and molar mass [kg/mol],
# kinetic diameter for the hard-sphere mean free path [m].
GASES = {
    "N2": {"refractivity_stp": 2.98e-4, "molar_mass": 28.0134e-3, "diameter": 3.75e-10},
    "H2": {"refractivity_stp": 1.39e-4, "molar_mass": 2.016e-3, "diameter": 2.89e-10},
}


def gas_density(p, T, gas):
    """Ideal-gas density [kg/m^3]."""
    return p * GASES[gas]["molar_mass"] / (R_GAS * T)


def refractivity(rho, gas):
    """Gladstone-Dale: n - 1 is proportional to density."""
    rho_stp = gas_density(P_STP, T_STP, gas)
    return GASES[gas]["refractivity_stp"] * rho / rho_stp


def mean_free_path(p, T, gas):
    """Hard-sphere mean free path lambda = k T / (sqrt(2) pi d^2 p) [m]."""
    d = GASES[gas]["diameter"]
    return K_B * T / (np.sqrt(2) * np.pi * d**2 * p)


def abel_matrix(r_edges, y):
    """Matrix A with P(y) = A @ f for a piecewise-constant radial profile f.

    For an annulus [r1, r2] with value f, the chord at height y crosses a
    length 2 (sqrt(r2^2 - y^2) - sqrt(max(r1, y)^2 - y^2)), which is exact
    for piecewise-constant profiles (no singular integrand to discretise).
    """
    r1 = r_edges[:-1][None, :]
    r2 = r_edges[1:][None, :]
    yy = np.abs(np.asarray(y))[:, None]
    outer = np.sqrt(np.clip(r2**2 - yy**2, 0, None))
    inner = np.sqrt(np.clip(np.maximum(r1, yy) ** 2 - yy**2, 0, None))
    return 2.0 * (outer - inner)


def project(field_xr, r_edges, y):
    """Line-of-sight integral of an axisymmetric field f(x, r) -> P(x, y)."""
    return field_xr @ abel_matrix(r_edges, y).T


@dataclass
class SchlierenSetup:
    """Z-type Schlieren system with a knife edge cutting half the source image."""

    focal_length: float = 1.0        # second mirror focal length f2 [m]
    cutoff: float = 0.5e-3           # unobstructed source-image height a [m]
    full_well_fraction: float = 0.5  # background at 50% of full well (I/I0 = 1)
    full_well: float = 20000.0       # camera full well [electrons]
    read_noise: float = 5.0          # [electrons rms]

    @property
    def background_electrons(self):
        return self.full_well_fraction * self.full_well

    def intensity(self, deflection):
        """Relative intensity I/I0 for a deflection angle [rad] across the knife edge.

        I/I0 = 1 + f2 * eps / a, limited by fully blocked (0) and fully passed
        (h/a = 2 for a half cut-off) source images.
        """
        return np.clip(1.0 + self.focal_length * deflection / self.cutoff, 0.0, 2.0)

    def sensitivity(self):
        """d(I/I0)/d(eps) in the linear range [1/rad]."""
        return self.focal_length / self.cutoff

    def noise_sigma(self, i_rel):
        """Per-pixel noise in units of I0 (shot + read noise)."""
        e = self.background_electrons * i_rel
        return np.sqrt(e + self.read_noise**2) / self.background_electrons

    def camera(self, i_rel, rng=None, frames=1):
        """Noisy image in units of I0, averaged over `frames` exposures."""
        rng = np.random.default_rng(0) if rng is None else rng
        e = self.background_electrons * i_rel
        shots = rng.poisson(np.clip(e, 0, None) * frames) + rng.normal(0, self.read_noise * np.sqrt(frames), e.shape)
        return shots / (frames * self.background_electrons)


def schlieren_image(rho_rel, x, r_edges, y, nozzle_d, p_ambient, T_ambient, gas, setup: SchlierenSetup):
    """Synthetic Schlieren image of a jet solution.

    Args:
        rho_rel: density in units of the ambient density, shape (nx, nr), on the
            solver grid in units of the nozzle diameter.
        x: axial cell centres [D]; r_edges: radial cell edges [D]; y: image rows [D].
        nozzle_d: nozzle diameter [m]; p_ambient [Pa]; T_ambient [K].
    Returns:
        (I/I0 noise-free image, deflection angle [rad]) with shape (nx, len(y)).
    The knife edge is vertical, so the image shows the axial gradient d/dx,
    which makes normal shocks such as the Mach disk stand out.
    """
    rho_abs = rho_rel * gas_density(p_ambient, T_ambient, gas)
    dn = refractivity(rho_abs, gas)
    path = project(dn, r_edges, y) * nozzle_d            # integral of (n-1) dz [m]
    eps = np.gradient(path, x * nozzle_d, axis=0)         # deflection [rad]
    return setup.intensity(eps), eps
