import numpy as np
import jax
from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.utilities import measure_time
from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_jax,
    lippmann_schwinger_cuda,
)


jax.config.update("jax_enable_x64", True)


def initialise_material(grid_spec, fibre_radius=0.2, dtype=jnp.float64):
    """Material coefficients lambda and mu evaluated at voxel centres

    Returns two arrays of shape (nx,ny,nz)

    :arg grid_spec: grid specification
    :arg fibre_radius: radius of fibre
    :arg dtype: data type
    """
    X, Y, Z = np.meshgrid(
        grid_spec.Lx / grid_spec.nx * (1 / 2 + np.arange(grid_spec.nx)),
        grid_spec.Ly / grid_spec.ny * (1 / 2 + np.arange(grid_spec.ny)),
        grid_spec.Lz / grid_spec.nz * (1 / 2 + np.arange(grid_spec.nz)),
        indexing="ij",
    )
    mu = np.ones(shape=(grid_spec.nx, grid_spec.ny, grid_spec.nz)) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )

    lmbda = np.ones(shape=(grid_spec.nx, grid_spec.ny, grid_spec.nz)) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )
    return jnp.array(mu, dtype=dtype), jnp.array(lmbda, dtype=dtype)


# Domain size in all three spatial direction
Lx = 1.0
Ly = 1.0
Lz = 1.0
# Number of grid cells in all three spatial directions
nx = 64
ny = 64
nz = 64

dtype = jnp.float32
rtol = 1e-20
atol = 1e-4
depth = 0

grid_spec = GridSpec(nx, ny, nz, Lx, Ly, Lz)
mu, lmbda = initialise_material(grid_spec, dtype=dtype)
E_mean = jnp.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)


with measure_time("evaluation [Jax]"):
    epsilon, sigma, iter = lippmann_schwinger_jax(
        lmbda, mu, E_mean, grid_spec, maxiter=32, depth=depth, rtol=rtol, atol=atol
    )
    epsilon.block_until_ready()

with measure_time("evaluation [CUDA]"):
    epsilon, sigma, iter = lippmann_schwinger_cuda(
        lmbda, mu, E_mean, grid_spec, maxiter=32, rtol=rtol, atol=atol, verbose=2
    )

with measure_time("gradient"):
    grad_epsilon = jax.jacfwd(lippmann_schwinger_jax, argnums=[2])
    dg = grad_epsilon(lmbda, mu, E_mean, grid_spec, depth=depth, rtol=rtol, atol=atol)
    dg[0][0].block_until_ready()
