import numpy as np
from contextlib import contextmanager
import time
from collections import namedtuple
from matplotlib import pyplot as plt
import jax

from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.fourier import get_xi
from jaxmaterials.solver.lippmann_schwinger import (
    relative_divergence,
    relative_divergence_fourier,
    lippmann_schwinger_jax,
    lippmann_schwinger_cuda,
)


jax.config.update("jax_enable_x64", True)
devices = jax.devices()
print(f"Available Jax devices: {devices}")


#######
#######
#######
@contextmanager
def measure_time(label="Elapsed Time"):
    """Measure the time it takes to execute a block of code

    :arg label: label for the time measurement
    """
    t_start = time.perf_counter()
    try:
        yield
    finally:
        t_finish = time.perf_counter()
        t_elapsed = t_finish - t_start
        print(f"time [{label}] = {t_elapsed:8.2f} s")


def initialise_material(grid_spec, sphere_radius=0.2, lmbda_lst=[1, 2], mu_lst=[1, 2], dtype=jnp.float64):
    """Material coefficients lambda and mu evaluated at voxel centres

    Returns two arrays of shape (N_0,N_1,N_2)

    :arg grid_spec: namedtuple with grid specification
    :arg fibre_radius: radius of fibre
    :arg dtype: data type
    """
    shape = (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    dimension = (grid_spec.Lx, grid_spec.Ly, grid_spec.Lz)
    X, Y, Z = np.meshgrid(
        *[L/n * (1 / 2 + np.arange(n)) for (n, L) in zip(shape, dimension)],
        indexing="ij",
    )
    
    idx = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + 5*(Z - 0.5) ** 2 < sphere_radius**2

    mu = np.zeros(shape=shape)
    mu[idx] = mu_lst[0]
    mu[~idx] = mu_lst[1]

    lmbda = np.zeros(shape=shape)
    lmbda[idx] = lmbda_lst[0]
    lmbda[~idx] = lmbda_lst[1]
    
    return jnp.array(lmbda, dtype=dtype), jnp.array(mu, dtype=dtype)

##
##
def get_meanstress(
        lmbda,
        mu,
        epsilon_bar,
        grid_spec,
        rtol=1e-6,
        atol=1e-20,
        depth=0,
        maxiter=32,
        dtype=jnp.float32,
        ):
    epsilon, sigma, iter = lippmann_schwinger_jax(
        lmbda,
        mu,
        epsilon_bar,
        grid_spec,
        rtol=1e-6,
        atol=1e-20,
        depth=0,
        maxiter=32,
        dtype=jnp.float32,
        )
    return jnp.mean(sigma, axis=(1,2,3))

###############
###############
###############


# Domain size in all three spatial direction
Lx = 1.0
Ly = 2.0
Lz = 1.5
# Number of grid cells in all three spatial directions
Nx = 20
Ny = 40
Nz = 30

dtype = jnp.float64
tolerance = 1.0e-9

grid_spec = GridSpec(Nx, Ny, Nz,Lx, Ly, Lz)
lmbda, mu = initialise_material(grid_spec, sphere_radius=0.5, lmbda_lst=[100, 2], mu_lst=[50, 1], dtype=dtype)
E_mean = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(lmbda[5])
ax[1].imshow(lmbda[10])
ax[2].imshow(lmbda[19])
plt.show()

with measure_time("evaluation"):
    #epsilon, sigma, iter = lippmann_schwinger(
    #    lmbda, mu, E_mean, grid_spec, tolerance=tolerance
    #)
    epsilon, sigma, iter = lippmann_schwinger_jax(
        lmbda, mu, E_mean, grid_spec, rtol=1e-6, atol=1e-20, depth=3, maxiter=100, dtype=jnp.float32
    )
    jax.block_until_ready(iter)
print(f"finished evaluation after {iter:5d} iterations")

with measure_time("gradient"):
    grad_epsilon = jax.jacfwd(get_meanstress, argnums=[2])
    dg = grad_epsilon(lmbda, mu, E_mean, grid_spec, rtol=1e-6, atol=1e-20, depth=0, maxiter=100, dtype=jnp.float32)
    jax.block_until_ready(dg)


plotting = False
if plotting:
    X, Y = np.meshgrid(Lx / Nx * (0.5 + np.arange(Nx)), Ly / Ny * (0.5 + np.arange(Ny)))
    # Plot strain field
    plt.clf()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.contourf(X, Y, epsilon[0, :, :, 30])
    #plt.savefig("epsilon.pdf", bbox_inches="tight")
    # Plot stress field
    plt.clf()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.contourf(X, Y, sigma[0, :, :, 0])
    #plt.savefig("sigma.pdf", bbox_inches="tight")
    plt.show()

for row in dg[0]:
    print(" ".join(f"{val:3}" for val in row))
