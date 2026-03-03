import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
import jax
from jaxmaterials.utilities import measure_time

# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jaxmaterials.linear_elasticity import *


def initialise_material(grid_spec, fibre_radius=0.2, dtype=jnp.float64):
    """Material coefficients lambda and mu evaluated at voxel centres

    Returns two arrays of shape (N_0,N_1,N_2)

    :arg grid_spec: namedtuple with grid specification
    :arg fibre_radius: radius of fibre
    :arg dtype: data type
    """
    X, Y, Z = np.meshgrid(
        *[L/float(n) * (1 / 2 + np.arange(n)) for (n, L) in zip(grid_spec.N, grid_spec.L)],
        indexing="ij",
    )
    mu = np.ones(shape=grid_spec.N) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )

    lmbda = np.ones(shape=grid_spec.N) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )
    return jnp.array(mu, dtype=dtype), jnp.array(lmbda, dtype=dtype)


devices = jax.devices()
print(f"Available Jax devices: {devices}")

GridSpec = namedtuple("GridSpec", ["N", "L"])

# Domain size in all three spatial direction
Lx = 1.0
Ly = 1.0
Lz = 1.0
# Number of grid cells in all three spatial directions
Nx = 64
Ny = 64
Nz = 64

dtype = jnp.float32
tolerance = 1.0e-3
depth = 0

grid_spec = GridSpec(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz))
xi = get_xizero(grid_spec)
mu, lmbda = initialise_material(grid_spec, dtype=dtype)
E_mean = jnp.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)

with measure_time("evaluation [Jax]"):
    epsilon, sigma, iter = lippmann_schwinger(
        lmbda, mu, E_mean, grid_spec, maxiter=32, depth=depth, tolerance=tolerance
    )
    epsilon.block_until_ready()

with measure_time("evaluation [CUDA]"):
    epsilon, sigma, iter = lippmann_schwinger_cuda(
        lmbda, mu, E_mean, grid_spec, maxiter=32, atol=tolerance, verbose=2
    )

with measure_time("gradient"):
    grad_epsilon = jax.jacfwd(lippmann_schwinger, argnums=[2])
    dg = grad_epsilon(lmbda, mu, E_mean, grid_spec, depth=depth, tolerance=tolerance)
    dg[0][0].block_until_ready()

plotting = False
if plotting:
    X, Y = np.meshgrid(Lx / Nx * (0.5 + np.arange(Nx)), Ly / Ny * (0.5 + np.arange(Ny)))
    # Plot strain field
    plt.clf()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.contourf(X, Y, epsilon[0, :, :, 0])
    plt.savefig("epsilon.pdf", bbox_inches="tight")
    # Plot stress field
    plt.clf()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.contourf(X, Y, sigma[0, :, :, 0])
    plt.savefig("sigma.pdf", bbox_inches="tight")
