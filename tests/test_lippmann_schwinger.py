import pytest
import numpy as np

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.fourier import get_xi
from jaxmaterials.solver.lippmann_schwinger import (
    relative_divergence,
    relative_divergence_fourier,
)


@pytest.fixture
def grid_spec():

    # Domain size in all three spatial direction
    Lx = 1.2
    Ly = 0.8
    Lz = 0.9
    # Number of grid cells in all three spatial directions
    nx = 64
    ny = 48
    nz = 32

    return GridSpec(nx, ny, nz, Lx, Ly, Lz)


def test_relative_convergence(grid_spec):
    """Verify that the relative divergence is computed consistently in real space
    and Fourier space"""

    xi = get_xi(grid_spec)
    rng = np.random.default_rng(seed=8741823)
    sigma = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz))
    sigma_hat = np.fft.fftn(sigma, axes=[-3, -2, -1])
    rel_div_real = relative_divergence(sigma, grid_spec)
    rel_div_fourier = relative_divergence_fourier(sigma_hat, xi, grid_spec)
    tolerance = 1.0e-5
    assert abs((rel_div_real - rel_div_fourier) / rel_div_real) < tolerance
