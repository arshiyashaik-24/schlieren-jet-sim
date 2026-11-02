"""Optics validation: Abel projection, Gladstone-Dale, knife edge, camera noise."""
import numpy as np
import pytest

from jetlab.optics import (
    SchlierenSetup, abel_matrix, gas_density, mean_free_path, project, refractivity,
)


def test_abel_projection_of_gaussian_matches_analytic():
    # f(r) = exp(-r^2/s^2)  ->  P(y) = sqrt(pi) s exp(-y^2/s^2)
    s = 0.5
    edges = np.linspace(0, 4, 801)
    rc = 0.5 * (edges[1:] + edges[:-1])
    y = np.linspace(-1.5, 1.5, 61)
    P = project(np.exp(-(rc**2) / s**2)[None, :], edges, y)[0]
    exact = np.sqrt(np.pi) * s * np.exp(-(y**2) / s**2)
    assert np.max(np.abs(P - exact)) < 2e-3 * exact.max()


def test_abel_of_uniform_disk_is_chord_length():
    edges = np.linspace(0, 1, 101)
    y = np.array([0.0, 0.3, 0.99, 1.2])
    P = abel_matrix(edges, y).sum(axis=1)
    expected = 2 * np.sqrt(np.clip(1 - y**2, 0, None))
    assert P == pytest.approx(expected, abs=1e-12)


def test_air_refractivity_at_room_conditions():
    rho = gas_density(101325, 293.15, "N2")
    assert refractivity(rho, "N2") == pytest.approx(2.78e-4, rel=0.01)


def test_hydrogen_is_weaker_than_nitrogen_at_equal_pressure():
    p, T = 100.0, 293.15
    ratio = refractivity(gas_density(p, T, "H2"), "H2") / refractivity(gas_density(p, T, "N2"), "N2")
    assert ratio == pytest.approx(1.39 / 2.98, rel=1e-6)


def test_mean_free_path_of_nitrogen_at_one_atmosphere():
    # Textbook hard-sphere value is ~60-70 nm at room temperature.
    assert 50e-9 < mean_free_path(101325, 293.15, "N2") < 80e-9


def test_knife_edge_response_is_linear_then_saturates():
    s = SchlierenSetup(focal_length=1.0, cutoff=0.5e-3)
    assert s.intensity(0.0) == 1.0
    assert s.intensity(1e-5) == pytest.approx(1.02)
    assert s.intensity(1.0) == 2.0 and s.intensity(-1.0) == 0.0


def test_camera_noise_matches_model():
    s = SchlierenSetup()
    img = s.camera(np.ones((400, 400)), rng=np.random.default_rng(1))
    assert img.std() == pytest.approx(s.noise_sigma(1.0), rel=0.02)
    img16 = s.camera(np.ones((400, 400)), rng=np.random.default_rng(2), frames=16)
    assert img16.std() == pytest.approx(s.noise_sigma(1.0) / 4, rel=0.03)
