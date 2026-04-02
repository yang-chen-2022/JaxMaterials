"""Functionality for computations in Fourier space"""

import numpy as np
from jax import numpy as jnp

__all__ = ["get_xizero", "get_xi", "fourier_solve"]


def get_xi(grid_spec, dtype=jnp.float64):
    """Construct the un-normalised frequency vectors

    Let k = (k_0,k_1,k_2) with k_d = 0,1,...,N_d-1 be a three-dimensional Fourier index.

    The normalised momentum vector is xi_d = 2 pi k_d / N_d, with 0 <= xi_0 < 2pi

    For a given k we then have that

    tilde(xi)_0 = 2/h_0 * sin(xi_0/2) * cos(xi_1/2) * cos(xi_2/2)
    tilde(xi)_1 = 2/h_1 * cos(xi_0/2) * sin(xi_1/2) * cos(xi_2/2)
    tilde(xi)_2 = 2/h_2 * cos(xi_0/2) * cos(xi_1/2) * sin(xi_2/2)

    This function returns a tensor of shape (3,N_0,N_1,N_2) which contains
    the normalised xi^0 = tilde(xi) for all Fourier modes.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    # Normalised momentum vectors in all three spatial directions
    K = [
        2 * np.pi * np.arange(n) / n for n in (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    ]
    # Grid with normalised momentum vectors
    xi = np.meshgrid(*K, indexing="ij")
    # Grid with tilde(xi)
    xi = np.stack(
        [
            2
            * grid_spec.nx
            / grid_spec.Lx
            * np.sin(xi[0] / 2)
            * np.cos(xi[1] / 2)
            * np.cos(xi[2] / 2),
            2
            * grid_spec.ny
            / grid_spec.Ly
            * np.cos(xi[0] / 2)
            * np.sin(xi[1] / 2)
            * np.cos(xi[2] / 2),
            2
            * grid_spec.nz
            / grid_spec.Lz
            * np.cos(xi[0] / 2)
            * np.cos(xi[1] / 2)
            * np.sin(xi[2] / 2),
        ]
    )
    return xi.astype(dtype)


def get_xizero(grid_spec, dtype=jnp.float64):
    """Construct the normalised frequency vectors

    Let k = (k_0,k_1,k_2) with k_d = 0,1,...,N_d-1 be a three-dimensional Fourier index.

    The normalised momentum vector is xi_d = 2 pi k_d / N_d, with 0 <= xi_0 < 2pi

    For a given k we then have that

    tilde(xi)_0 = 2/h_0 * sin(xi_0/2) * cos(xi_1/2) * cos(xi_2/2)
    tilde(xi)_1 = 2/h_1 * cos(xi_0/2) * sin(xi_1/2) * cos(xi_2/2)
    tilde(xi)_2 = 2/h_2 * cos(xi_0/2) * cos(xi_1/2) * sin(xi_2/2)

    This function returns a tensor of shape (3,nx,ny,nz) which contains
    the normalised xi^0 = tilde(xi) / ||tilde(xi)|| for all Fourier modes.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    xi_tilde = get_xi(grid_spec, dtype=dtype)
    # Normalise tilde(xi) to obtain xi^0
    xi_nrm = np.linalg.norm(xi_tilde, axis=0)
    xi_nrm[xi_nrm < 1.0e-12] = 1  # avoid division by zero
    return (xi_tilde / xi_nrm).astype(dtype)


def get_laplacian(grid_spec, dtype=jnp.float64):
    """Construct the laplacian operator in Fourier space

    This function returns a tensor of shape (1,N_0,N_1,N_2) which contains
    the laplacian operator (xixi) in fourier space.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    dx = grid_spec.Lx / grid_spec.nx
    dy = grid_spec.Ly / grid_spec.ny
    dz = grid_spec.Lz / grid_spec.nz
    # Normalised momentum vectors in all three spatial directions
    K = [
        2 * np.pi * np.arange(n) / n for n in (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    ]
    # Grid with normalised momentum vectors
    xi = np.meshgrid(*K, indexing="ij")
    # Grid with xi*xi (laplacian in Fourier space)
    xixi = 2.* (np.cos(xi[0]) - 1.) / dx**2 \
         + 2.* (np.cos(xi[1]) - 1.) / dy**2 \
         + 2.* (np.cos(xi[2]) - 1.) / dz**2
    return xixi[np.newaxis,...].astype(dtype)



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
            * (xizero[2, ...] * tau_hat[4, ...] + xizero[1, ...] * tau_hat[3]),
            xizero[1, ...] ** 2 * tau_hat[1, ...]
            + xizero[1, ...]
            * (xizero[2, ...] * tau_hat[5, ...] + xizero[0, ...] * tau_hat[3]),
            xizero[2, ...] ** 2 * tau_hat[2, ...]
            + xizero[2, ...]
            * (xizero[1, ...] * tau_hat[5, ...] + xizero[0, ...] * tau_hat[4]),
            1
            / 2
            * (
                xizero[0, ...] * xizero[1, ...] * (tau_hat[0, ...] + tau_hat[1, ...])
                + (xizero[0, ...] ** 2 + xizero[1, ...] ** 2) * tau_hat[3, ...]
                + xizero[2, ...]
                * (xizero[0, ...] * tau_hat[5, ...] + xizero[1, ...] * tau_hat[4, ...])
            ),
            1
            / 2
            * (
                xizero[0, ...] * xizero[2, ...] * (tau_hat[0, ...] + tau_hat[2, ...])
                + (xizero[0, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[4, ...]
                + xizero[1, ...]
                * (xizero[0, ...] * tau_hat[5, ...] + xizero[2, ...] * tau_hat[3, ...])
            ),
            1
            / 2
            * (
                xizero[1, ...] * xizero[2, ...] * (tau_hat[1, ...] + tau_hat[2, ...])
                + (xizero[1, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[5, ...]
                + xizero[0, ...]
                * (xizero[1, ...] * tau_hat[4, ...] + xizero[2, ...] * tau_hat[3, ...])
            ),
        ]
    )
    Xi = jnp.stack(
        [
            xizero[0, ...] ** 2,
            xizero[1, ...] ** 2,
            xizero[2, ...] ** 2,
            xizero[0, ...] * xizero[1, ...],
            xizero[0, ...] * xizero[2, ...],
            xizero[1, ...] * xizero[2, ...],
        ]
    )
    Xi_dot_tau = (
        xizero[0, ...] ** 2 * tau_hat[0, ...]
        + xizero[1, ...] ** 2 * tau_hat[1, ...]
        + xizero[2, ...] ** 2 * tau_hat[2, ...]
        + 2
        * (
            xizero[0, ...] * xizero[1, ...] * tau_hat[3, ...]
            + xizero[0, ...] * xizero[2, ...] * tau_hat[4, ...]
            + xizero[1, ...] * xizero[2, ...] * tau_hat[5, ...]
        )
    )
    epsilon_hat_B = Xi * Xi_dot_tau
    return (
        1 / mu0 * (-epsilon_hat_A + (lmbda0 + mu0) / (lmbda0 + 2 * mu0) * epsilon_hat_B)
    )
