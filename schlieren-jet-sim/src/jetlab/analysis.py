"""Locating the Mach disk: from the flow solution (ground truth) and from images."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def crist_correlation(p0_over_pa):
    """Empirical Mach disk location x_M / D = 0.67 sqrt(p0 / pa).

    Crist, Sherman & Glass, AIAA J. 4(1), 1966, for sonic nozzles over a wide
    range of stagnation-to-ambient pressure ratios.
    """
    return 0.67 * np.sqrt(p0_over_pa)


def ashkenas_sherman_mach(x_over_d, g=1.4):
    """Centreline Mach number in the free-expansion core of a sonic jet.

    M = A (x/D - x0/D)^(g-1) - (g+1)/(2(g-1)) / (A (x/D - x0/D)^(g-1)),
    with A = 3.65 and x0/D = 0.40 for g = 1.4 (Ashkenas & Sherman, 1966),
    a far-field fit to method-of-characteristics results upstream of the
    Mach disk, so it is not accurate close to the nozzle (x/D below ~1.5).
    It does not depend on the pressure ratio, so it checks the expansion
    independently of where the shock ends up.
    """
    a, x0 = 3.65, 0.40
    t = a * np.clip(np.asarray(x_over_d) - x0, 1e-9, None) ** (g - 1)
    return t - (g + 1) / (2 * (g - 1)) / t


def _subcell_peak(values, i):
    """Parabolic refinement of a discrete maximum at index i (in index units)."""
    if 0 < i < len(values) - 1:
        a, b, c = values[i - 1], values[i], values[i + 1]
        den = a - 2 * b + c
        if den != 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def mach_disk_from_flow(rho_axis, mach_axis, x):
    """Mach disk position from the solution on the axis.

    The disk is the first normal shock that decelerates the supersonic core
    to subsonic speed. Walking downstream from the nozzle, find the first
    point where the axial Mach number falls from above 1.5 to below 1; the
    shock is the steepest density rise within a few cells of that point.
    (Taking the steepest rise anywhere can pick a later shock cell.)
    """
    grad = np.gradient(rho_axis, x)
    was_fast = np.maximum.accumulate(mach_axis > 1.5)
    drops = np.where(was_fast & (mach_axis < 1.0))[0]
    if len(drops) == 0:
        return float("nan")
    k = int(drops[0])
    lo, hi = max(k - 4, 1), min(k + 2, len(x) - 1)
    i = lo + int(np.argmax(grad[lo:hi]))
    return float(np.interp(_subcell_peak(grad, i), np.arange(len(x)), x))


def mach_disk_from_image(image, x, y, x_min=0.5, smooth_cells=1.5):
    """Mach disk position from a Schlieren image (knife edge sensitive to d/dx).

    Averages the rows within a quarter diameter of the axis, smooths along x,
    and returns the brightest feature downstream of `x_min`: the compression
    at the Mach disk deflects light towards the passed side of the knife edge.
    """
    rows = np.abs(y) <= 0.25
    profile = image[:, rows].mean(axis=1) - 1.0
    profile = gaussian_filter1d(profile, smooth_cells)
    valid = x >= x_min
    score = np.where(valid, profile, -np.inf)
    i = int(np.argmax(score))
    refined = _subcell_peak(np.where(valid, profile, profile.min()), i)
    return float(np.interp(refined, np.arange(len(x)), x)), profile
