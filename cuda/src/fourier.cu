/* Implementation of fourier_solve_device.hh */
#include "fourier.hh"

/* kernel to initialize Fourier vectors */
__global__ void initialize_xi_kernel(float *dev_xi, const GridSpec grid_spec)
{
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
        float xi_0_half = M_PI * k_a / nx;
        float xi_1_half = M_PI * k_b / ny;
        float xi_2_half = M_PI * k_c / nz;
        dev_xi[FIDX(nx, ny, nz, 0, k_a, k_b, k_c)] = two_hx_inv * sin(xi_0_half) * cos(xi_1_half) * cos(xi_2_half);
        dev_xi[FIDX(nx, ny, nz, 1, k_a, k_b, k_c)] = two_hy_inv * cos(xi_0_half) * sin(xi_1_half) * cos(xi_2_half);
        dev_xi[FIDX(nx, ny, nz, 2, k_a, k_b, k_c)] = two_hz_inv * cos(xi_0_half) * cos(xi_1_half) * sin(xi_2_half);
    }
}

/* Initialize Fourier vectors*/
void initialize_xi(float *dev_xi,
                   const GridSpec grid_spec)
{
    int nx = grid_spec.nx;
    int ny = grid_spec.ny;
    int nz = grid_spec.nz;
    dim3 grid((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    initialize_xi_kernel<<<grid, block>>>(dev_xi, grid_spec);
}

/* kernel to initialize Fourier vectors */
__global__ void initialize_xizero_kernel(float *dev_xi_zero, const GridSpec grid_spec)
{
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
        float xi_0_half = M_PI * k_a / nx;
        float xi_1_half = M_PI * k_b / ny;
        float xi_2_half = M_PI * k_c / nz;
        float tilde_xi_0 = two_hx_inv * sin(xi_0_half) * cos(xi_1_half) * cos(xi_2_half);
        float tilde_xi_1 = two_hy_inv * cos(xi_0_half) * sin(xi_1_half) * cos(xi_2_half);
        float tilde_xi_2 = two_hz_inv * cos(xi_0_half) * cos(xi_1_half) * sin(xi_2_half);
        float tilde_xi_nrm = sqrt(tilde_xi_0 * tilde_xi_0 + tilde_xi_1 * tilde_xi_1 + tilde_xi_2 * tilde_xi_2);
        // Avoid division by zero
        const float tolerance = 1.E-6;
        if (tilde_xi_nrm < tolerance)
            tilde_xi_nrm = 1.0;
        dev_xi_zero[FIDX(nx, ny, nz, 0, k_a, k_b, k_c)] = tilde_xi_0 / tilde_xi_nrm;
        dev_xi_zero[FIDX(nx, ny, nz, 1, k_a, k_b, k_c)] = tilde_xi_1 / tilde_xi_nrm;
        dev_xi_zero[FIDX(nx, ny, nz, 2, k_a, k_b, k_c)] = tilde_xi_2 / tilde_xi_nrm;
    }
}

/* Initialize Fourier vectors*/
void initialize_xizero(float *dev_xi_zero,
                       const GridSpec grid_spec)
{
    int nx = grid_spec.nx;
    int ny = grid_spec.ny;
    int nz = grid_spec.nz;
    dim3 grid((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    initialize_xizero_kernel<<<grid, block>>>(dev_xi_zero, grid_spec);
}

/* kernel to initialize Fourier vectors on host */
void initialize_xizero_host(float *xi_zero,
                            const GridSpec grid_spec)
{
    int nx = grid_spec.nx;
    int ny = grid_spec.ny;
    int nz = grid_spec.nz;
    float two_hx_inv = 2 * grid_spec.nx / grid_spec.Lx;
    float two_hy_inv = 2 * grid_spec.ny / grid_spec.Ly;
    float two_hz_inv = 2 * grid_spec.nz / grid_spec.Lz;
    for (int k_c = 0; k_c < nz; ++k_c)
        for (int k_b = 0; k_b < ny; ++k_b)
            for (int k_a = 0; k_a < nx; ++k_a)
            {
                float xi_0_half = M_PI * k_a / nx;
                float xi_1_half = M_PI * k_b / ny;
                float xi_2_half = M_PI * k_c / nz;
                float tilde_xi_0 = two_hx_inv * sin(xi_0_half) * cos(xi_1_half) * cos(xi_2_half);
                float tilde_xi_1 = two_hy_inv * cos(xi_0_half) * sin(xi_1_half) * cos(xi_2_half);
                float tilde_xi_2 = two_hz_inv * cos(xi_0_half) * cos(xi_1_half) * sin(xi_2_half);
                float tilde_xi_nrm = sqrt(tilde_xi_0 * tilde_xi_0 + tilde_xi_1 * tilde_xi_1 + tilde_xi_2 * tilde_xi_2);
                // Avoid division by zero
                const float tolerance = 1.E-6;
                if (tilde_xi_nrm < tolerance)
                    tilde_xi_nrm = 1.0;
                xi_zero[FIDX(nx, ny, nz, 0, k_a, k_b, k_c)] = tilde_xi_0 / tilde_xi_nrm;
                xi_zero[FIDX(nx, ny, nz, 1, k_a, k_b, k_c)] = tilde_xi_1 / tilde_xi_nrm;
                xi_zero[FIDX(nx, ny, nz, 2, k_a, k_b, k_c)] = tilde_xi_2 / tilde_xi_nrm;
            }
}

/* kernel for Fourier solve in homogeneous isotropic reference material */
__global__ void fourier_solve_kernel(cufftComplex *dev_tau_hat, cufftComplex *dev_epsilon_hat,
                                     float *dev_xi_zero,
                                     const float C_A, const float C_B,
                                     const int ncells)
{
    int ell = blockDim.x * blockIdx.x + threadIdx.x;
    float xi[3];
    cufftComplex tau_hat[6];
    cufftComplex epsilon_hat[6];
    if (ell < ncells)
    {
        // copy into temporary arrays
        for (int mu = 0; mu < 3; ++mu)
            xi[mu] = dev_xi_zero[mu * ncells + ell];
        for (int mu = 0; mu < 6; ++mu)
            tau_hat[mu] = dev_tau_hat[mu * ncells + ell];
        cufftComplex rho;
        rho.x = xi[0] * xi[0] * tau_hat[0].x +
                xi[1] * xi[1] * tau_hat[1].x +
                xi[2] * xi[2] * tau_hat[2].x +
                2 * (xi[0] * xi[1] * tau_hat[3].x +
                     xi[0] * xi[2] * tau_hat[4].x +
                     xi[1] * xi[2] * tau_hat[5].x);
        rho.y = xi[0] * xi[0] * tau_hat[0].y +
                xi[1] * xi[1] * tau_hat[1].y +
                xi[2] * xi[2] * tau_hat[2].y +
                2 * (xi[0] * xi[1] * tau_hat[3].y +
                     xi[0] * xi[2] * tau_hat[4].y +
                     xi[1] * xi[2] * tau_hat[5].y);
        epsilon_hat[0].x = C_A * xi[0] * (xi[0] * tau_hat[0].x + xi[2] * tau_hat[4].x + xi[1] * tau_hat[3].x) +
                           C_B * rho.x * xi[0] * xi[0];
        epsilon_hat[0].y = C_A * xi[0] * (xi[0] * tau_hat[0].y + xi[2] * tau_hat[4].y + xi[1] * tau_hat[3].y) +
                           C_B * rho.y * xi[0] * xi[0];
        epsilon_hat[1].x = C_A * xi[1] * (xi[1] * tau_hat[1].x + xi[2] * tau_hat[5].x + xi[0] * tau_hat[3].x) +
                           C_B * rho.x * xi[1] * xi[1];
        epsilon_hat[1].y = C_A * xi[1] * (xi[1] * tau_hat[1].y + xi[2] * tau_hat[5].y + xi[0] * tau_hat[3].y) +
                           C_B * rho.y * xi[1] * xi[1];
        epsilon_hat[2].x = C_A * xi[2] * (xi[2] * tau_hat[2].x + xi[1] * tau_hat[5].x + xi[0] * tau_hat[4].x) +
                           C_B * rho.x * xi[2] * xi[2];
        epsilon_hat[2].y = C_A * xi[2] * (xi[2] * tau_hat[2].y + xi[1] * tau_hat[5].y + xi[0] * tau_hat[4].y) +
                           C_B * rho.y * xi[2] * xi[2];
        epsilon_hat[3].x = 0.5 * C_A * (xi[0] * xi[1] * (tau_hat[0].x + tau_hat[1].x) + (xi[0] * xi[0] + xi[1] * xi[1]) * tau_hat[3].x + xi[2] * (xi[0] * tau_hat[5].x + xi[1] * tau_hat[4].x)) + C_B * rho.x * xi[0] * xi[1];
        epsilon_hat[3].y = 0.5 * C_A * (xi[0] * xi[1] * (tau_hat[0].y + tau_hat[1].y) + (xi[0] * xi[0] + xi[1] * xi[1]) * tau_hat[3].y + xi[2] * (xi[0] * tau_hat[5].y + xi[1] * tau_hat[4].y)) + C_B * rho.y * xi[0] * xi[1];
        epsilon_hat[4].x = 0.5 * C_A * (xi[0] * xi[2] * (tau_hat[0].x + tau_hat[2].x) + (xi[0] * xi[0] + xi[2] * xi[2]) * tau_hat[4].x + xi[1] * (xi[0] * tau_hat[5].x + xi[2] * tau_hat[3].x)) + C_B * rho.x * xi[0] * xi[2];
        epsilon_hat[4].y = 0.5 * C_A * (xi[0] * xi[2] * (tau_hat[0].y + tau_hat[2].y) + (xi[0] * xi[0] + xi[2] * xi[2]) * tau_hat[4].y + xi[1] * (xi[0] * tau_hat[5].y + xi[2] * tau_hat[3].y)) + C_B * rho.y * xi[0] * xi[2];
        epsilon_hat[5].x = 0.5 * C_A * (xi[1] * xi[2] * (tau_hat[1].x + tau_hat[2].x) + (xi[1] * xi[1] + xi[2] * xi[2]) * tau_hat[5].x + xi[0] * (xi[1] * tau_hat[4].x + xi[2] * tau_hat[3].x)) + C_B * rho.x * xi[1] * xi[2];
        epsilon_hat[5].y = 0.5 * C_A * (xi[1] * xi[2] * (tau_hat[1].y + tau_hat[2].y) + (xi[1] * xi[1] + xi[2] * xi[2]) * tau_hat[5].y + xi[0] * (xi[1] * tau_hat[4].y + xi[2] * tau_hat[3].y)) + C_B * rho.y * xi[1] * xi[2];
        // copy back into solution vector
        for (int mu = 0; mu < 6; ++mu)
            dev_epsilon_hat[mu * ncells + ell] = epsilon_hat[mu];
    }
}

/* Fourier solve for homogeneous isotropic reference material */
void fourier_solve_device(cufftComplex *dev_tau_hat, cufftComplex *dev_epsilon_hat,
                          float *dev_xi_zero,
                          const float lambda_0, const float mu_0,
                          const GridSpec grid_spec)
{
    int ncells = grid_spec.number_of_cells();
    const int nblocks = (ncells + BLOCKSIZE - 1) / BLOCKSIZE;
    const float C_A = -1.0 / mu_0;
    const float C_B = (lambda_0 + mu_0) / (mu_0 * (lambda_0 + 2 * mu_0));
    fourier_solve_kernel<<<nblocks, BLOCKSIZE>>>(dev_tau_hat, dev_epsilon_hat, dev_xi_zero,
                                                 C_A, C_B, ncells);
}