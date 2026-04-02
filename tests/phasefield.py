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
    phase_field_solve,
    elasticity_solve,
    compute_strain_energy,
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


def initialise_material(grid_spec, shape="homogeneous", center=None, size=0.2, 
                        mu_mat=1, lmbda_mat=1, 
                        mu_inc=0.5, lmbda_inc=0.5, margin_width= 0.05,
                        add_crack=True, crack_length=0.5,
                        dtype=jnp.float64):
    """Material coefficients lambda and mu evaluated at voxel centres

    
    :arg shape: "sphere", "cuboid", or "cyclinder"(Parallel to z-axis)
    :arg center: Tuple of (x,y,z) coordinates. Defaults to grid center.
    :arg size: Radius for sphere, or edge length(s) for cuboid.
    :arg mu_mat/lmbda_mat: Lamé parameters for the background matrix.
    :arg mu_inc/lmbda_inc: Lamé parameters for the inclusion (fibre).
    """

    Lx = grid_spec.Lx
    Ly = grid_spec.Ly
    Lz = grid_spec.Lz

    N = [grid_spec.nx, grid_spec.ny, grid_spec.nz]
    h = [grid_spec.Lx/grid_spec.nx, grid_spec.Ly/grid_spec.ny, grid_spec.Lz/grid_spec.nz]

    # Find the center if one isnt inputted
    if center is None:
        cx, cy, cz = Lx / 2, Ly / 2, Lz / 2
    else:
        cx, cy, cz = center

    
    X, Y, Z = np.meshgrid(
        *[h * (1 / 2 + np.arange(n)) for (n, h) in zip(N, h)],
        indexing="ij",
    )
    
    
    if shape == "sphere":
        # The size acts as the radius
        idx = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 < size**2
        
    elif shape == "cuboid":
        # Check if size is a single number (cube) or a list of 3 numbers (rectangle)
        if isinstance(size, (int, float)):
            sx, sy, sz = size, size, size
        else:
            sx, sy, sz = size
            
        idx = (np.abs(X - cx) < sx/2) & (np.abs(Y - cy) < sy/2) & (np.abs(Z - cz) < sz/2)

    elif shape == "cylinder":

        # Check if 'size' is a single number (continuous) or a tuple (chopped short fiber)
        if isinstance(size, (int, float)):
            # Continuous fiber: We only restrict X and Y. It runs infinitely along Z.
            idx = (X - cx)**2 + (Y - cy)**2 < size**2
        else:
            # Chopped fiber: size = (radius, length)
            r, length = size
            idx = ((X - cx)**2 + (Y - cy)**2 < r**2) & (np.abs(Z - cz) < length/2)

    elif shape == "homogeneous":
        # An empty mask (no inclusion at all)
        idx = np.zeros_like(X, dtype=bool)
        
        
    else:
        raise ValueError(f"Unknown shape requested: {shape}")


  
    #True if inside use _inc else use _mat
    mu = np.where(idx, mu_inc, mu_mat)
    
    lmbda = np.where(idx, lmbda_inc, lmbda_mat)


    if margin_width is not None:
    
        margin_bound = (X < margin_width) | (X > (Lx - margin_width))
        
        # Override the Lame parameters with 0.0 in the margin regions
        mu = np.where(margin_bound, 0.0, mu)
        lmbda = np.where(margin_bound, 0.0, lmbda)
    
    if add_crack:
        
        crack_end_x = margin_width + crack_length
        
        h_y = h[1]
        crack_y_max = Ly / 2
        crack_y_min = (Ly / 2) - h_y 
        
        crack_bound = (X <= crack_end_x) & (Y > crack_y_min) & (Y < crack_y_max)
        
        # Override the Lame parameters with 0.0 in the crack region
        mu = np.where(crack_bound, 0.0, mu)
        lmbda = np.where(crack_bound, 0.0, lmbda)
        
    return jnp.array(mu, dtype=dtype), jnp.array(lmbda, dtype=dtype)
    

    
###############
###############
###############


# Domain size in all three spatial direction
Lx = 1.0
Ly = 1.0
Lz = 0.015
# Number of grid cells in all three spatial directions
Nx = 55
Ny = 50
Nz = 1

dtype = jnp.float64
tolerance = 1.0e-9

grid_spec = GridSpec(Nx, Ny, Nz,Lx, Ly, Lz)

lmbda_solid = 121.15  #GPa [kN/mm2]   
mu_solid    = 80.77  
mu, lmbda   = initialise_material(grid_spec, shape="homogeneous",
                                    mu_mat=mu_solid, lmbda_mat=lmbda_solid, dtype=dtype)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(lmbda)
ax[1].imshow(mu)
plt.show()

with measure_time("Total time: "):
    Emean_steps = [jnp.array([0.0, eps_yy, 0.0, 0.0, 0.0, 0.0], dtype=dtype) 
                  for eps_yy in np.linspace(0, 0.01, 51)]
    save_steps = [0, 50, 100, 150, 199]

    gc = 2.7e-3 #TODO: make this a field
    lc = 0.015 #TODO: make this a field
    k_stab = 1e-6

    # Staggered scheme for solving elasticity + phase-field equations
    d = jnp.zeros((grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype)
    HH = jnp.zeros((grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype)

    dfield = {}
    epsfield = {}
    sigfield = {}
    
    sig_steps = []
    eps_steps = []

    for step, E_mean in enumerate(Emean_steps):
        print(f"Time Step {step} ")
        with measure_time("phase-field solve"):
            d, iter_pf = phase_field_solve(HH, d, gc, lc, grid_spec, tolerance, maxiter=10000, dtype=dtype)
            jax.block_until_ready(iter_pf)
            print(f"  PF solve converged in {iter_pf} iterations.")

        with measure_time("elasticity solve"):
            epsilon, sigma, iter = elasticity_solve(
                lmbda, mu, E_mean, d, k_stab, grid_spec, rtol=1e-6, atol=1e-20, depth=3, maxiter=1000, dtype=jnp.float32
            )
            jax.block_until_ready(iter)
            print(f"  Elasticity solve converged in {iter} iterations.")

        #  Save & display
        sigAV  = np.array([np.mean(sigma[i]) for i in range(6)])
        sig_steps.append(sigAV)

        epsAV = np.array([np.mean(epsilon[i]) for i in range(6)])
        eps_steps.append(epsAV)

        # update the history field
        psi = compute_strain_energy(lmbda, mu, epsilon)
        HH = jnp.maximum(HH, psi)
        jax.block_until_ready(HH)
        
        if step in save_steps:
            dfield[step] = d
            epsfield[step] = epsilon
            sigfield[step] = sigma

            fig, ax = plt.subplots(2, 3)
            ax[0,0].imshow(psi)
            ax[0,1].imshow(HH)
            ax[0,2].imshow(d)
            ax[1,0].imshow(epsilon[1])
            ax[1,2].imshow(sigma[1])
            ax[1,2].imshow(lmbda)
            plt.title(f"Step {step}")
            plt.show
        
eps_steps = np.array(eps_steps)
sig_steps = np.array(sig_steps)

print(eps_steps[:,1])
print(sig_steps[:,1])
plt.figure()
plt.plot(eps_steps[:,1], sig_steps[:,1], marker='o')
plt.show()