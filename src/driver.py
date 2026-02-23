import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
import jax
from jaxmaterials.utilities import measure_time
import ctypes

# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jaxmaterials.linear_elasticity import *


devices = jax.devices()
print(f"Available Jax devices: {devices}")

GridSpec = namedtuple("GridSpec", ["N", "h"])

# Domain size in all three spatial direction
Lx = 1.0
Ly = 1.0
Lz = 1.0
# Number of grid cells in all three spatial directions
Nx = 64
Ny = 64
Nz = 64

dtype = jnp.float32
tolerance = 1.0e-4
depth = 0

grid_spec = GridSpec(N=(Nx, Ny, Nz), h=(Lx / Nx, Ly / Ny, Lz / Nz))
xi = get_xizero(grid_spec)
mu, lmbda = initialise_material(grid_spec, dtype=dtype)
E_mean = jnp.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)

with measure_time("evaluation"):
    epsilon, sigma, iter = lippmann_schwinger(
        lmbda, mu, E_mean, grid_spec, maxiter=32, depth=depth, tolerance=tolerance
    )
    epsilon.block_until_ready()
    # jax.profiler.save_device_memory_profile("memory.prof")
print(f"finished evaluation after {iter:5d} iterations")
with measure_time("gradient"):
    grad_epsilon = jax.jacfwd(lippmann_schwinger, argnums=[2])
    dg = grad_epsilon(lmbda, mu, E_mean, grid_spec, depth=depth, tolerance=tolerance)
    dg[0][0].block_until_ready()

# Load cuda library
lib = ctypes.CDLL("../cuda/build/lib/liblippmannschwinger.so")
lippmann_schwinger_cuda = lib.lippmann_schwinger_solve
lippmann_schwinger_cuda.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
]
cells = np.array([64, 64, 64], dtype=np.int32)
extents = np.array([1.0, 1.0, 1.0], dtype=np.float32)
with measure_time("cuda"):
    lippmann_schwinger_cuda(
        np.asarray(mu),
        np.asarray(lmbda),
        np.asarray(E_mean),
        np.asarray(epsilon),
        np.asarray(sigma),
        cells,
        extents,
    )

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
