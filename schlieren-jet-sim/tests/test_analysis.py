"""Mach disk detectors on synthetic profiles with a known answer."""
import numpy as np
import pytest

from jetlab.analysis import crist_correlation, mach_disk_from_flow, mach_disk_from_image


def test_crist_correlation_values():
    assert crist_correlation(10) == pytest.approx(2.119, abs=1e-3)


def test_flow_detector_finds_normal_shock_with_subcell_accuracy():
    x = np.linspace(0, 6, 193)
    x_shock = 2.345
    # Supersonic, expanding core up to the shock; subsonic after it.
    width = 0.03
    step = 0.5 * (1 + np.tanh((x - x_shock) / width))
    mach = 4.0 * (1 - step) + 0.4 * step
    rho = 0.2 * (1 - step) + 1.0 * step
    assert mach_disk_from_flow(rho, mach, x) == pytest.approx(x_shock, abs=0.25 * (x[1] - x[0]))


def test_flow_detector_ignores_compressions_in_subsonic_flow():
    x = np.linspace(0, 6, 193)
    mach = np.where(x < 2.0, 3.0, 0.5)
    rho = np.where(x < 2.0, 0.3, 1.0) + np.where(x > 4.0, 5.0, 0.0)  # larger jump, but subsonic
    assert mach_disk_from_flow(rho, mach, x) == pytest.approx(2.0, abs=0.05)


def test_flow_detector_picks_first_shock_not_strongest():
    # Two shock cells: the second terminating shock is stronger, but the
    # Mach disk is the first one.
    x = np.linspace(0, 8, 257)
    mach = np.where(x < 2.0, 3.0, np.where(x < 4.0, 0.6, np.where(x < 5.0, 2.5, 0.5)))
    rho = np.where(x < 2.0, 0.3, np.where(x < 4.0, 1.0, np.where(x < 5.0, 0.2, 2.0)))
    assert mach_disk_from_flow(rho, mach, x) == pytest.approx(2.0, abs=0.05)


def test_image_detector_on_noisy_bright_line():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 193)
    y = np.linspace(-2, 2, 129)
    img = 1 + 0.3 * np.exp(-((x[:, None] - 3.1) ** 2) / (2 * 0.04**2)) * np.ones_like(y)[None, :]
    img += rng.normal(0, 0.05, img.shape)
    x_det, _ = mach_disk_from_image(img, x, y)
    assert x_det == pytest.approx(3.1, abs=0.03)
