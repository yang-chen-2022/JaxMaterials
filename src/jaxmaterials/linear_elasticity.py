import numpy as np
import jax

from jax import numpy as jnp

__all__ = ["lippmann_schwinger"]


def get_xizero(grid_spec, dtype=jnp.float64):
    """Construct the normalised frequency vectors

    Let k = (k_0,k_1,k_2) with k_d = 0,1,...,N_d-1 be a three-dimensional Fourier index.

    The normalised momentum vector is xi_d = 2 pi k_d / N_d, with 0 <= xi_0 < 2pi

    For a given k we then have that

    tilde(xi)_0 = 2/h_0 * sin(xi_0/2) * cos(xi_1/2) * cos(xi_2/2)
    tilde(xi)_1 = 2/h_1 * cos(xi_0/2) * sin(xi_1/2) * cos(xi_2/2)
    tilde(xi)_2 = 2/h_2 * cos(xi_0/2) * cos(xi_1/2) * sin(xi_2/2)

    This function returns a tensor of shape (3,N_0,N_1,N_2) which contains
    the normalised xi^0 = tilde(xi) / ||tilde(xi)|| for all Fourier modes.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    # Normalised momentum vectors in all three spatial directions
    K = [2 * np.pi * np.arange(n) / n for n in grid_spec.N]
    # Grid with normalised momentum vectors
    xi = np.meshgrid(*K, indexing="ij")
    h = grid_spec.h
    # Grid with tilde(xi)
    xi_tilde = np.stack(
        [
            2 / h[0] * np.sin(xi[0] / 2) * np.cos(xi[1] / 2) * np.cos(xi[2] / 2),
            2 / h[1] * np.cos(xi[0] / 2) * np.sin(xi[1] / 2) * np.cos(xi[2] / 2),
            2 / h[2] * np.cos(xi[0] / 2) * np.cos(xi[1] / 2) * np.sin(xi[2] / 2),
        ]
    )
    # Normalise tilde(xi) to obtain xi^0
    xi_nrm = np.linalg.norm(xi_tilde, axis=0)
    xi_nrm[xi_nrm < 1.0e-12] = 1  # avoid division by zero
    return (xi_tilde / xi_nrm).astype(dtype)


def initialise_material(grid_spec, fibre_radius=0.2, dtype=jnp.float64):
    """Material coefficients lambda and mu evaluated at voxel centres

    Returns two arrays of shape (N_0,N_1,N_2)

    :arg grid_spec: namedtuple with grid specification
    :arg fibre_radius: radius of fibre
    :arg dtype: data type
    """
    X, Y, Z = np.meshgrid(
        *[h * (1 / 2 + np.arange(n)) for (n, h) in zip(grid_spec.N, grid_spec.h)],
        indexing="ij",
    )
    mu = np.ones(shape=grid_spec.N) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )

    lmbda = np.ones(shape=grid_spec.N) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )
    return jnp.array(mu, dtype=dtype), jnp.array(lmbda, dtype=dtype)


def backward_derivative(g, grid_spec, direction):
    """Compute the backward derivative in a specified direction

    :arg g: function to take the derivative of. Assumed to be of shape (*,N_0,N_1,N_2)
    :arg grid_spec: namedtuple with grid specification
    :arg direction: direction in which to take the derivative: 0, 1 or 2
    """
    if direction == 0:
        dg = (
            g
            + jnp.roll(g, 1, axis=-2)
            + jnp.roll(g, 1, axis=-1)
            + jnp.roll(g, (1, 1), axis=(-2, -1))
            - jnp.roll(g, 1, axis=-3)
            - jnp.roll(g, (1, 1), axis=(-3, -2))
            - jnp.roll(g, (1, 1), axis=(-3, -1))
            - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
        )
    elif direction == 1:
        dg = (
            g
            + jnp.roll(g, 1, axis=-3)
            + jnp.roll(g, 1, axis=-1)
            + jnp.roll(g, (1, 1), axis=(-3, -1))
            - jnp.roll(g, 1, axis=-2)
            - jnp.roll(g, (1, 1), axis=(-3, -2))
            - jnp.roll(g, (1, 1), axis=(-2, -1))
            - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
        )
    elif direction == 2:
        dg = (
            g
            + jnp.roll(g, 1, axis=-3)
            + jnp.roll(g, 1, axis=-2)
            + jnp.roll(g, (1, 1), axis=(-3, -2))
            - jnp.roll(g, 1, axis=-1)
            - jnp.roll(g, (1, 1), axis=(-3, -1))
            - jnp.roll(g, (1, 1), axis=(-2, -1))
            - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
        )

    return 1 / (4 * grid_spec.h[direction]) * dg


def backward_divergence(sigma, grid_spec):
    """Compute backward derivative of symmetric 3x3 tensor sigma_{ij}

    The componets of the tensor are assumed to be represented in vector form
    using Voigt notation:

    (sigma_{00}, sigma_{11}, sigma_{22}, sigma_{12}, sigma_{02}, sigma_{01})

    Returns a tensor of shape (3,N_0,N_1,N_2) with the three components of the
    divergence vector:

    [ dsigma_{00}/dx_0 + dsigma_{01}/dx_1 + dsigma_{02}/dx_2 ]
    [ dsigma_{10}/dx_0 + dsigma_{11}/dx_1 + dsigma_{12}/dx_2 ]
    [ dsigma_{20}/dx_0 + dsigma_{21}/dx_1 + dsigma_{22}/dx_2 ]

    :arg sigma: tensor representation using Voigt notation
    :arg grid_spec: namedtuple with grid specification
    """
    return jnp.stack(
        [
            backward_derivative(sigma[0, ...], grid_spec, 0)
            + backward_derivative(sigma[5, ...], grid_spec, 1)
            + backward_derivative(sigma[4, ...], grid_spec, 2),
            backward_derivative(sigma[5, ...], grid_spec, 0)
            + backward_derivative(sigma[1, ...], grid_spec, 1)
            + backward_derivative(sigma[3, ...], grid_spec, 2),
            backward_derivative(sigma[4, ...], grid_spec, 0)
            + backward_derivative(sigma[3, ...], grid_spec, 1)
            + backward_derivative(sigma[2, ...], grid_spec, 2),
        ]
    )


def fourier_solve(tau_hat, lmbda0, mu0, xizero):
    """Solve residual equation for reference material in Fourier space

    Computes hat(epsilon)_{kl} = -Gamma^0_{klij} hat(tau)_{ij}

    :arg tau_hat: The residual hat(tau) in Fourier space
    :arg lmbda0: coefficient lambda^0 of homogeneous reference material
    :arg mu0: coefficient mu^0 of homogeneous reference material
    :arg xizero: Normalised momentum vectors
    """
    epsilon_hat_A = jnp.stack(
        [
            xizero[0, ...] ** 2 * tau_hat[0, ...]
            + xizero[0, ...]
            * (xizero[2, ...] * tau_hat[4, ...] + xizero[1, ...] * tau_hat[5]),
            xizero[1, ...] ** 2 * tau_hat[1, ...]
            + xizero[1, ...]
            * (xizero[2, ...] * tau_hat[3, ...] + xizero[0, ...] * tau_hat[5]),
            xizero[2, ...] ** 2 * tau_hat[2, ...]
            + xizero[2, ...]
            * (xizero[1, ...] * tau_hat[3, ...] + xizero[0, ...] * tau_hat[4]),
            1
            / 2
            * (
                xizero[1, ...] * xizero[2, ...] * (tau_hat[1, ...] + tau_hat[2, ...])
                + (xizero[1, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[3, ...]
                + xizero[0, ...]
                * (xizero[1, ...] * tau_hat[4, ...] + xizero[2, ...] * tau_hat[5, ...])
            ),
            1
            / 2
            * (
                xizero[0, ...] * xizero[2, ...] * (tau_hat[0, ...] + tau_hat[2, ...])
                + (xizero[0, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[4, ...]
                + xizero[1, ...]
                * (xizero[0, ...] * tau_hat[3, ...] + xizero[2, ...] * tau_hat[5, ...])
            ),
            1
            / 2
            * (
                xizero[0, ...] * xizero[1, ...] * (tau_hat[0, ...] + tau_hat[1, ...])
                + (xizero[0, ...] ** 2 + xizero[1, ...] ** 2) * tau_hat[5, ...]
                + xizero[2, ...]
                * (xizero[0, ...] * tau_hat[3, ...] + xizero[1, ...] * tau_hat[4, ...])
            ),
        ]
    )
    Xi = jnp.stack(
        [
            xizero[0, ...] ** 2,
            xizero[1, ...] ** 2,
            xizero[2, ...] ** 2,
            xizero[1, ...] * xizero[2, ...],
            xizero[0, ...] * xizero[2, ...],
            xizero[0, ...] * xizero[1, ...],
        ]
    )
    Xi_dot_tau = (
        xizero[0, ...] ** 2 * tau_hat[0, ...]
        + xizero[1, ...] ** 2 * tau_hat[1, ...]
        + xizero[2, ...] ** 2 * tau_hat[2, ...]
        + 2
        * (
            xizero[1, ...] * xizero[2, ...] * tau_hat[3, ...]
            + xizero[0, ...] * xizero[2, ...] * tau_hat[4, ...]
            + xizero[0, ...] * xizero[1, ...] * tau_hat[5, ...]
        )
    )
    epsilon_hat_B = Xi * Xi_dot_tau
    return (
        1 / mu0 * (-epsilon_hat_A + (lmbda0 + mu0) / (lmbda0 + 2 * mu0) * epsilon_hat_B)
    )


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


def relative_divergence(sigma, grid_spec):
    """Compute ratio divergence div(sigma)_i = d sigma_{ij} / dx_j  and <||div(sigma)||^2>^{1/2}

    :arg sigma: stress
    :arg grid_spec: grid specification
    """
    dV = grid_spec.h[0] * grid_spec.h[1] * grid_spec.h[2]
    dsigma = backward_divergence(sigma, grid_spec)
    dsigma_nrm = jnp.sqrt(jnp.sum(dsigma**2) * dV)
    # Compute average <sigma> and norm ||<sigma>||
    sigma_avg = jnp.sum(sigma * dV, axis=[1, 2, 3])
    sigma_avg_nrm = jnp.sqrt(
        (jnp.sum(sigma_avg[:3] ** 2) + 2 * jnp.sum(sigma_avg[3:] ** 2))
    )
    return dsigma_nrm / sigma_avg_nrm


@jax.jit(static_argnames=["grid_spec", "tolerance", "depth"])
def lippmann_schwinger(
    lmbda, mu, E_mean, grid_spec, tolerance=1e-8, depth=0, maxiter=32
):
    """Lippmann Schwinger iteration with Anderson acceleration for linear elasticity

    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter lambda
    :arg E_mean: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg tolerance: tolerance on (relative) stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    """
    # reference values of Lame paraeter
    mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
    lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=E_mean.dtype)
    # storage for solution and residual, arrays of shape (d+1,6,Nx,Ny,Nz)
    epsilon = jnp.zeros(
        (depth + 1, 6) + grid_spec.N,
        dtype=E_mean.dtype,
    )
    epsilon = epsilon.at[0, ...].set(jnp.expand_dims(E_mean, [1, 2, 3]))
    residual = jnp.zeros((depth + 1, 6) + grid_spec.N, dtype=epsilon.dtype)
    # Anderson matrix and vectors
    A_anderson = jnp.eye(depth + 1, dtype=epsilon.dtype)
    u_rhs = jnp.zeros(depth + 1, dtype=epsilon.dtype)
    sigma = compute_sigma(lmbda, mu, epsilon[0, ...])

    def exit_condition(state):
        """Check exit condition

        Check whether <||div(sigma)||> / ||<sigma>|| > tolerance or iter < maxiter

        :arg state: current iteration state (epsilon, residual, sigma, A, iter)
        """
        epsilon, residual, sigma, A_anderson, u_rhs, iter = state
        rel_error = relative_divergence(sigma, grid_spec)
        return (rel_error > tolerance) & (iter < maxiter)

    def loop_body(state):
        """Update strain, residual and stress according to update rule

        :arg state: current iteration state (epsilon, residual, sigma, A, iter)
        """
        epsilon, residual, sigma, A_anderson, u_rhs, iter = state
        # Fourier transform sigma
        sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
        # Solve reference problem hat{epsilon}_{kl} = -Gamma^0_{klij} hat{tau}_{ij}
        r_hat = fourier_solve(sigma_hat, lmbda0, mu0, xizero)
        r = jnp.real(jnp.fft.ifftn(r_hat, axes=[-3, -2, -1]))
        residual = jnp.roll(residual, 1, axis=0)
        residual = residual.at[0, ...].set(r)
        A_anderson = jnp.roll(A_anderson, (1, 1), axis=(0, 1))
        dotproduct_scaling = jnp.array(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=epsilon.dtype
        )
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
        iter += 1
        return (epsilon, residual, sigma, A_anderson, u_rhs, iter)

    epsilon, residual, sigma, A_anderson, u_rhs, iter = jax.lax.while_loop(
        exit_condition,
        loop_body,
        init_val=(epsilon, residual, sigma, A_anderson, u_rhs, 0),
    )

    return epsilon[0, ...], sigma, iter
