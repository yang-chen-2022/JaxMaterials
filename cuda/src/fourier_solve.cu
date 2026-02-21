/* Implementation of fourier_solve.hh */
#include "fourier_solve.hh"

/** @brief Initialize Fourier vectors
 *
 * @param[in] grid_spec grid specification
 */
__global__ void initialize_xizero_kernel(float *dev_xi_zero_0,
                                         float *dev_xi_zero_1,
                                         float *dev_xi_zero_2,
                                         const GridSpec grid_spec)
{
    const float pi = 3.1415926535897932384626433;
    int nx = grid_spec.nx;
    int ny = grid_spec.ny;
    int nz = grid_spec.nz;
    float two_hx_inv = 2 * grid_spec.nx / grid_spec.Lx;
    float two_hy_inv = 2 * grid_spec.ny / grid_spec.Ly;
    float two_hz_inv = 2 * grid_spec.nz / grid_spec.Lz;
    int k_a = blockDim.x * blockIdx.x + threadIdx.x;
    int k_b = blockDim.y * blockIdx.y + threadIdx.y;
    int k_c = blockDim.z * blockIdx.z + threadIdx.z;
    if ((k_a < nx) && (k_b < ny) && (k_c < nz))
    {
        float xi_0_half = pi * k_a / nx;
        float xi_1_half = pi * k_b / ny;
        float xi_2_half = pi * k_c / nz;
        float tilde_xi_0 = two_hx_inv * sin(xi_0_half) * cos(xi_1_half) * cos(xi_2_half);
        float tilde_xi_1 = two_hy_inv * cos(xi_0_half) * sin(xi_1_half) * cos(xi_2_half);
        float tilde_xi_2 = two_hz_inv * cos(xi_0_half) * cos(xi_1_half) * sin(xi_2_half);
        float tilde_xi_nrm = sqrt(tilde_xi_0 * tilde_xi_0 + tilde_xi_1 * tilde_xi_1 + tilde_xi_2 * tilde_xi_2);
        dev_xi_zero_0[IDX(nx, ny, nz, k_a, k_b, k_c)] = tilde_xi_0 / tilde_xi_nrm;
        dev_xi_zero_1[IDX(nx, ny, nz, k_a, k_b, k_c)] = tilde_xi_1 / tilde_xi_nrm;
        dev_xi_zero_2[IDX(nx, ny, nz, k_a, k_b, k_c)] = tilde_xi_1 / tilde_xi_nrm;
    }
}

void initialize_xizero(float *dev_xi_zero_0,
                       float *dev_xi_zero_1,
                       float *dev_xi_zero_2,
                       const GridSpec grid_spec)
{
    int nx = grid_spec.nx;
    int ny = grid_spec.ny;
    int nz = grid_spec.nz;
    dim3 grid((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    initialize_xizero_kernel<<<grid, block>>>(dev_xi_zero_0, dev_xi_zero_1, dev_xi_zero_2, grid_spec);
}

__global__ void fourier_solve_kernel(float *sigma_hat, float *r_hat, const GridSpec grid_spec)
{
    int nx = grid_spec.nx;
    int ny = grid_spec.ny;
    int nz = grid_spec.nz;
    int a = blockDim.x * blockIdx.x + threadIdx.x;
    int b = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;
    if ((a < nx) && (b < ny) && (c < nz))
    {
        sigma_hat[IDX(nx, ny, nz, a, b, c)] = 0;
        r_hat[IDX(nx, ny, nz, a, b, c)] = 0;
    }
}

void fourier_solve(float *dev_sigma_hat, float *dev_r_hat, const GridSpec grid_spec)
{
    int nx = grid_spec.nx;
    int ny = grid_spec.ny;
    int nz = grid_spec.nz;
    dim3 grid((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    fourier_solve_kernel<<<grid, block>>>(dev_sigma_hat, dev_r_hat, grid_spec);
}