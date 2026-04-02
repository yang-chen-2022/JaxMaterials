"""Lippmann Schwinger solver with Anderson acceleration"""

import ctypes
import numpy as np
import jax
from jax import numpy as jnp
from jaxmaterials.solver.derivatives import backward_divergence
from jaxmaterials.solver.fourier import get_xizero, get_xi, get_laplacian, fourier_solve

from jaxmaterials.utilities import voigt_to_tensor, tensor_to_voigt

__all__ = [
    "relative_divergence",
    "relative_divergence_fourier",
    "lippmann_schwinger_jax",
    "lippmann_schwinger_cuda",
]


def compute_sigma(lmbda, mu, epsilon):
    """Compute stress from strain

    Returns sigma_{ij} = C_{ijkl}*epsilon_{kl}

    :arg lmbda: Lame parameter lambda
    :arg mu: Lame parameter mu
    :arg epsilon: strain field
    """
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = 2 * mu * epsilon + lmbda * jnp.stack(
        3 * [tr_epsilon] + 3 * [jnp.zeros(epsilon.shape[-3:], dtype=epsilon.dtype)]
    )
    return sigma


def compute_sigma_damaged(lmbda, mu, epsilon, d=None, k=1e-12):
    """Compute stress with asymmetric degradation."""
    eps_tensor = voigt_to_tensor(epsilon)
    
    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)
    tr_eps_minus = jnp.minimum(tr_eps, 0.0)
    
    # Get eigenvalues n eigenvectors
    eigvals, eigvecs = jnp.linalg.eigh(eps_tensor)
    
    eigvals_plus = jnp.maximum(eigvals, 0.0)
    eigvals_minus = jnp.minimum(eigvals, 0.0)
    
    # Reconstruct the positive and negative strain tensors (eps_plus / eps_minus)
    # This uses einsum to do: V * Lambda_plus * V^T across the entire 3D grid instantly
    eps_plus_tensor = jnp.einsum('...ia,...a,...ja->...ij', eigvecs, eigvals_plus, eigvecs)
    eps_minus_tensor = jnp.einsum('...ia,...a,...ja->...ij', eigvecs, eigvals_minus, eigvecs)
    
    # Convert back to Voigt notation for the stress equation
    eps_plus_v = tensor_to_voigt(eps_plus_tensor)
    eps_minus_v = tensor_to_voigt(eps_minus_tensor)
    
    # Identity matrix in Voigt notation for the Trace operation
    I_voigt = jnp.stack([
        jnp.ones_like(tr_eps_plus), jnp.ones_like(tr_eps_plus), jnp.ones_like(tr_eps_plus), 
        jnp.zeros_like(tr_eps_plus), jnp.zeros_like(tr_eps_plus), jnp.zeros_like(tr_eps_plus)
    ], axis=0)
    
    # Calculate pure tension stress and pure compression stress
    sigma_plus = lmbda * tr_eps_plus * I_voigt + 2.0 * mu * eps_plus_v
    sigma_minus = lmbda * tr_eps_minus * I_voigt + 2.0 * mu * eps_minus_v
    
    # Apply damage degradation (g_d) ONLY to the tension (positive) stress
    if d is not None:
        g_d = (1.0 - d)**2 + k
        return g_d * sigma_plus + sigma_minus
    
    return sigma_plus + sigma_minus


def compute_strain_energy(lmbda, mu, epsilon):
    """Compute the ONLY the positive/ tensile elastic strain energy to drive the fracture.
        
        epsilon should be of shape (6, Nx, Ny, Nz) - [11,22,33,12,13,23]
        """
    # 1. Convert to 3x3 tensor
    eps_tensor = voigt_to_tensor(epsilon)
    
    # 2. Get trace and split into positive part
    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)
    
    # 3. Calculate eigenvalues using JAX's eigh function
    eigvals = jnp.linalg.eigvalsh(eps_tensor)
    
    # 4. Filter only the positive eigenvalues
    eigvals_plus = jnp.maximum(eigvals, 0.0)
    eps_sq_plus = jnp.sum(eigvals_plus**2, axis=-1)
    
    # 5. Compute only the tensile energy (psi_plus)
    psi_plus = 0.5 * lmbda * (tr_eps_plus**2) + mu * eps_sq_plus
    
    return psi_plus



def relative_divergence(sigma, grid_spec):
    """Compute ratio of the norm of div(sigma) and the norm of the average sigma

    :arg sigma: stress
    :arg grid_spec: grid specification
    """
    dsigma = backward_divergence(sigma, grid_spec)
    dsigma_nrm = jnp.sqrt(jnp.sum(dsigma**2))
    sigma_avg = jnp.mean(sigma, axis=[1, 2, 3])
    sigma_avg_nrm = jnp.sqrt(
        jnp.sum(sigma_avg[:3] ** 2) + 2 * jnp.sum(sigma_avg[3:] ** 2)
    )
    return dsigma_nrm / (jnp.sqrt(grid_spec.number_of_voxels) * sigma_avg_nrm)


def relative_divergence_fourier(sigma_hat, xi, grid_spec):
    """Compute ratio of the norm of div(sigma) and the norm of the average sigma in Fourier space

    :arg sigma_hat: stress in Fourier space
    :arg xi: Fourier vectors
    :arg grid_spec: grid specification
    """
    dsigma_hat = jnp.stack(
        [
            xi[0, ...] * sigma_hat[0, ...]
            + xi[1, ...] * sigma_hat[3, ...]
            + xi[2, ...] * sigma_hat[4, ...],
            xi[0, ...] * sigma_hat[3, ...]
            + xi[1, ...] * sigma_hat[1, ...]
            + xi[2, ...] * sigma_hat[5, ...],
            xi[0, ...] * sigma_hat[4, ...]
            + xi[1, ...] * sigma_hat[5, ...]
            + xi[2, ...] * sigma_hat[2, ...],
        ]
    )
    dsigma_nrm = jnp.sqrt(jnp.sum(jnp.abs(dsigma_hat) ** 2))
    sigma_hat_zero = jnp.real(sigma_hat[:, 0, 0, 0])
    sigma_hat_zero_nrm = jnp.sqrt(
        jnp.sum(sigma_hat_zero[:3] ** 2) + 2 * jnp.sum(sigma_hat_zero[3:] ** 2)
    )
    return dsigma_nrm / sigma_hat_zero_nrm


@jax.jit(static_argnames=["grid_spec", "rtol", "atol", "depth", "maxiter", "dtype"])
def lippmann_schwinger_jax(
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
    """Lippmann Schwinger iteration with Anderson acceleration for linear elasticity

    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter lambda
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    # reference values of Lame paraeter
    mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
    lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=dtype)
    xi = get_xi(grid_spec, dtype=dtype)
    # storage for solution and residual, arrays of shape (d+1,6,Nx,Ny,Nz)
    epsilon = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz),
        dtype=dtype,
    )
    epsilon = epsilon.at[0, ...].set(
        jnp.expand_dims(jnp.astype(epsilon_bar, dtype), [1, 2, 3])
    )
    residual = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype
    )
    # Anderson matrix and vectors
    A_anderson = jnp.eye(depth + 1, dtype=dtype)
    u_rhs = jnp.zeros(depth + 1, dtype=dtype)
    sigma = compute_sigma(lmbda, mu, epsilon[0, ...])
    # Fourier transform sigma
    sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
    rel_error = relative_divergence_fourier(sigma_hat, xi, grid_spec)
    rel_error_0 = rel_error

    def exit_condition(state):
        """Check exit condition

        Let e^i = <||div(sigma^i)||> / ||<sigma^i>|| be the current normalised divergence

        This method checkes whether e^i < max (atol, rtol * e^0) or iter > maxiter

        :arg state: current iteration state (epsilon, residual, sigma, A, iter, rel_error, rel_error_0)
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = state
        return (rel_error > atol) & (rel_error > rtol * rel_error_0) & (iter < maxiter)

    def loop_body(state):
        """Update strain, residual and stress according to update rule

        :arg state: current iteration state (epsilon, residual, sigma,sigma_hat, A_anderson, iter, rel_error)
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = state
        # Solve reference problem hat{epsilon}_{kl} = -Gamma^0_{klij} hat{tau}_{ij}
        r_hat = fourier_solve(sigma_hat, lmbda0, mu0, xizero)
        r = jnp.real(jnp.fft.ifftn(r_hat, axes=[-3, -2, -1]))
        residual = jnp.roll(residual, 1, axis=0)
        residual = residual.at[0, ...].set(r)
        A_anderson = jnp.roll(A_anderson, (1, 1), axis=(0, 1))
        dotproduct_scaling = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=dtype)
        A_anderson = A_anderson.at[0, :].set(
            jnp.einsum("aijk,saijk,a->s", r, residual, dotproduct_scaling)
        )
        A_anderson = A_anderson.at[:, 0].set(A_anderson[0, :])
        u_rhs = jnp.roll(u_rhs, 1)
        u_rhs = u_rhs.at[0].set(1)
        v = jnp.linalg.solve(A_anderson, u_rhs)
        alpha = v / jnp.dot(v, u_rhs)
        epsilon_tilde = jnp.einsum("s,saijk", alpha, epsilon + residual)
        epsilon = jnp.roll(epsilon, 1, axis=0)
        epsilon = epsilon.at[0, ...].set(epsilon_tilde)
        sigma = compute_sigma(lmbda, mu, epsilon[0, ...])
        # Fourier transform sigma
        sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
        rel_error = relative_divergence_fourier(sigma_hat, xi, grid_spec)
        iter += 1
        return (
            epsilon,
            residual,
            sigma,
            sigma_hat,
            A_anderson,
            u_rhs,
            iter,
            rel_error,
        )

    epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = (
        jax.lax.while_loop(
            exit_condition,
            loop_body,
            init_val=(
                epsilon,
                residual,
                sigma,
                sigma_hat,
                A_anderson,
                u_rhs,
                0,
                rel_error_0,
            ),
        )
    )

    return epsilon[0, ...], sigma, iter


def lippmann_schwinger_cuda(
    lmbda, mu, epsilon_bar, grid_spec, rtol=1e-6, atol=1.0e-20, maxiter=32, verbose=0
):
    """Wrapper for CUDA Lippmann Schwinger solver

    Required access to compiled library liblippmannschwinger.so

    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter lambda
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg maxiter: maximal number of iterations
    :arg verbose: verbosity level
    """
    # Load cuda library
    try:
        lib = ctypes.CDLL("liblippmannschwinger.so")
    except Exception as exc:
        raise RuntimeError(
            "Unable to load cuda library liblippmannschwinger.so. Compile and check LD_LIBRARY_PATH."
        ) from exc
    cuda_code = lib.lippmann_schwinger_solve
    cuda_code.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda_code.restype = ctypes.c_int
    cells = np.array([grid_spec.nx, grid_spec.ny, grid_spec.nz], dtype=np.int32)
    extents = np.array([grid_spec.Lx, grid_spec.Ly, grid_spec.Lz], dtype=np.float32)
    epsilon = np.empty((6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32)
    sigma = np.empty((6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32)
    iter = cuda_code(
        np.asarray(mu),
        np.asarray(lmbda),
        np.asarray(epsilon_bar, dtype=np.float32),
        np.asarray(epsilon),
        np.asarray(sigma),
        cells,
        extents,
        rtol,
        atol,
        maxiter,
        verbose,
    )
    if iter == maxiter:
        raise RuntimeError(f"Solver failed to converge after {maxiter} iterations")
    return (
        epsilon,
        sigma,
        iter,
    )




@jax.jit(static_argnames=["grid_spec", "rtol", "atol", "depth", "maxiter", "dtype"])
def elasticity_solve(
    lmbda,
    mu,
    epsilon_bar,
    dvar,
    kk,
    grid_spec,
    rtol=1e-6,
    atol=1e-20,
    depth=0,
    maxiter=32,
    dtype=jnp.float32,
):
    """Lippmann Schwinger iteration with Anderson acceleration for linear elasticity

    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter lambda
    :arg epsilon_bar: mean value of epsilon
    :arg dvar: damage variable (field), (1, Nx, Ny, Nz)
    :arg kk: stability parameter
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    # reference values of Lame paraeter
    mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
    lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))  #TODO: make this global variables?
    
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=dtype)
    xi = get_xi(grid_spec, dtype=dtype)

    # storage for Anderson Acceleration: solution and residual, arrays of shape (d+1,6,Nx,Ny,Nz)
    epsilon = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz),
        dtype=dtype,
    )
    epsilon = epsilon.at[0, ...].set(
        jnp.expand_dims(jnp.astype(epsilon_bar, dtype), [1, 2, 3])
    )
    residual = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype
    )

    # Anderson matrix and vectors
    A_anderson = jnp.eye(depth + 1, dtype=dtype)
    u_rhs = jnp.zeros(depth + 1, dtype=dtype)

    # Initial stress and Fourier transform
    sigma = compute_sigma_damaged(lmbda, mu, epsilon[0, ...], dvar, kk)
    sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])

    # initial relative error
    rel_error = relative_divergence_fourier(sigma_hat, xi, grid_spec)
    rel_error_0 = rel_error

    def exit_condition(state):
        """Check exit condition

        Let e^i = <||div(sigma^i)||> / ||<sigma^i>|| be the current normalised divergence

        This method checkes whether e^i < max (atol, rtol * e^0) or iter > maxiter

        :arg state: current iteration state (epsilon, residual, sigma, A, iter, rel_error, rel_error_0)
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = state
        return (rel_error > atol) & (rel_error > rtol * rel_error_0) & (iter < maxiter)

    def loop_body(state):
        """Update strain, residual and stress according to update rule

        :arg state: current iteration state (epsilon, residual, sigma,sigma_hat, A_anderson, iter, rel_error)
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = state
        # Solve reference problem hat{epsilon}_{kl} = -Gamma^0_{klij} hat{tau}_{ij}
        r_hat = fourier_solve(sigma_hat, lmbda0, mu0, xizero)
        r = jnp.real(jnp.fft.ifftn(r_hat, axes=[-3, -2, -1]))
        residual = jnp.roll(residual, 1, axis=0)
        residual = residual.at[0, ...].set(r)
        A_anderson = jnp.roll(A_anderson, (1, 1), axis=(0, 1))
        dotproduct_scaling = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=dtype)
        A_anderson = A_anderson.at[0, :].set(
            jnp.einsum("aijk,saijk,a->s", r, residual, dotproduct_scaling)
        )
        A_anderson = A_anderson.at[:, 0].set(A_anderson[0, :])
        u_rhs = jnp.roll(u_rhs, 1)
        u_rhs = u_rhs.at[0].set(1)
        v = jnp.linalg.solve(A_anderson, u_rhs)
        alpha = v / jnp.dot(v, u_rhs)
        epsilon_tilde = jnp.einsum("s,saijk", alpha, epsilon + residual)
        epsilon = jnp.roll(epsilon, 1, axis=0)
        epsilon = epsilon.at[0, ...].set(epsilon_tilde)
        sigma = compute_sigma_damaged(lmbda, mu, epsilon[0, ...], dvar, kk)
        # Fourier transform sigma
        sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
        rel_error = relative_divergence_fourier(sigma_hat, xi, grid_spec)
        iter += 1
        return (
            epsilon,
            residual,
            sigma,
            sigma_hat,
            A_anderson,
            u_rhs,
            iter,
            rel_error,
        )

    epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = (
        jax.lax.while_loop(
            exit_condition,
            loop_body,
            init_val=(
                epsilon,
                residual,
                sigma,
                sigma_hat,
                A_anderson,
                u_rhs,
                0,
                rel_error_0,
            ),
        )
    )

    return epsilon[0, ...], sigma, iter

@jax.jit(static_argnames=["grid_spec", "tolerance", "maxiter", "dtype"])
def phase_field_solve(
    HH,
    d_old,
    gc,
    lc,
    grid_spec,
    tolerance=1e-6,
    maxiter=1000,
    dtype=jnp.float32,
):
    """Fixed-point iteration solver for phase-field problem (fracture)

    :arg HH: history strain energy (field), (1, Nx, Ny, Nz)
    :arg d_old: damage variable at previous time step (field), (1, Nx, Ny, Nz)
    :arg gc: fracture toughness (field), (1, Nx, Ny, Nz)
    :arg lc: regularisation length (field)), (1, Nx, Ny, Nz)
    :arg grid_spec: grid specification
    :arg tolerance: tolerance for convergence check
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    
    # Coefficients A^t_n and B^t_n
    A_n = 1.0 / (lc**2) + 2.0 * HH / (gc * lc)
    B_n = 2.0 * HH / (gc * lc)
    
    # Reference parameter A_0
    A0_n = 0.5 * (jnp.min(A_n) + jnp.max(A_n))

    # Laplacian operator in Fourier space
    #xixi = get_laplacian(grid_spec, dtype=dtype)
    xi = get_xi(grid_spec, dtype=dtype)
    xixi = xi[0]**2 + xi[1]**2 + xi[2]**2

    # 
    def exit_condition(state):
        """Check exit condition
        
        Check whether the residual ||chi^{k+1} - chi^k||_2 / ||chi^{k+1}||_2 > tolerance or iter > maxiter
        """
        d_k, chi_k, iter, residual = state
        return (residual > tolerance) & (iter < maxiter)
    
    def loop_body(state):
        """Update phase-field, polarisation field, and compute residual"""
        d_k, chi_k, iter, residual = state
        
        # FFT of the polarisation field: chi^k(x) -> hat{chi}^k(xi)
        chiF = jnp.fft.fftn(chi_k)
        
        # Compute new phase field in Fourier space
        dF = chiF / (A0_n + xixi)
        
        # Inverse FFT to get real-space phase field
        d_new = jnp.real(jnp.fft.ifftn(dF))
        
        # Update the polarisation field
        chi_new = B_n - (A_n - A0_n) * d_new
        
        # Convergence test based on the L-2 norm over the unit cell
        norm_diff = jnp.linalg.norm(chi_new - chi_k)
        norm_chi_new = jnp.linalg.norm(chi_new)
        
        residual = jnp.where(norm_chi_new > 1e-12, norm_diff / norm_chi_new, 0.0)
        
        return (d_new, chi_new, iter + 1, residual)
    
    # Initialising variables for the first iteration (at k=0)
    d_initial = d_old
    chi_initial = B_n - (A_n - A0_n) * d_initial

    # Set initial residual to 1
    initial_residual = jnp.array(1.0, dtype=dtype)

    # Execute the fixed-point iteration loop
    d_final, chi_final, iter_count, res_final = jax.lax.while_loop(
        exit_condition,
        loop_body,
        init_val=(d_initial, chi_initial, 0, initial_residual)
    )

    return d_final, iter_count


