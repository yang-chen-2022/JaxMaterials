/** @brief implementation of derivatives.hh */
#include "derivatives.hh"

/* ********************************** *
 *    naive CUDA kernels              *
 * ********************************** */

/* Kernel for backward derivative computation in x-direction */
__global__ void backward_derivative_x_kernel(float *u, float *du_dx,
                                             const GridSpec grid_spec)
{
  float h_inv = grid_spec.nx / grid_spec.Lx;
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz = grid_spec.nz;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;
  if ((a < nx) && (b < ny) && (c < nz))
  {
    float _du_dx = 0;
    // apply derivative
    _du_dx += u[IDX(nx, ny, nz, a, b, c)];
    _du_dx += u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, c)];
    _du_dx += u[IDX(nx, ny, nz, a, b, (c - 1 + nz) % nz)];
    _du_dx += u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, (c - 1 + nz) % nz)];
    _du_dx -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, c)];
    _du_dx -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny, c)];
    _du_dx -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, (c - 1 + nz) % nz)];
    _du_dx -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny, (c - 1 + nz) % nz)];
    //  Copy back to global memory
    du_dx[IDX(nx, ny, nz, a, b, c)] += 0.25 * h_inv * _du_dx;
  }
}

/* Kernel for backward derivative computation in y-direction */
__global__ void backward_derivative_y_kernel(float *u, float *du_dy,
                                             const GridSpec grid_spec)
{
  float h_inv = grid_spec.ny / grid_spec.Ly;
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz = grid_spec.nz;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;
  if ((a < nx) && (b < ny) && (c < nz))
  {
    float _du_dy = 0;
    // apply derivative
    _du_dy += u[IDX(nx, ny, nz, a, b, c)];
    _du_dy += u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, c)];
    _du_dy += u[IDX(nx, ny, nz, a, b, (c - 1 + nz) % nz)];
    _du_dy += u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, (c - 1 + nz) % nz)];
    _du_dy -= u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, c)];
    _du_dy -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny, c)];
    _du_dy -= u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, (c - 1 + nz) % nz)];
    _du_dy -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny, (c - 1 + nz) % nz)];
    //  Copy back to global memory
    du_dy[IDX(nx, ny, nz, a, b, c)] += 0.25 * h_inv * _du_dy;
  }
}

/* Kernel for backward derivative computation in y-direction */
__global__ void backward_derivative_z_kernel(float *u, float *du_dz,
                                             const GridSpec grid_spec)
{
  float h_inv = grid_spec.nz / grid_spec.Lz;
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz = grid_spec.nz;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;
  if ((a < nx) && (b < ny) && (c < nz))
  {
    float _du_dz = 0;
    // apply derivative
    _du_dz += u[IDX(nx, ny, nz, a, b, c)];
    _du_dz += u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, c)];
    _du_dz += u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, c)];
    _du_dz += u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny, c)];
    _du_dz -= u[IDX(nx, ny, nz, a, b, (c - 1 + nz) % nz)];
    _du_dz -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, (c - 1 + nz) % nz)];
    _du_dz -= u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, (c - 1 + nz) % nz)];
    _du_dz -= u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny, (c - 1 + nz) % nz)];
    //  Copy back to global memory
    du_dz[IDX(nx, ny, nz, a, b, c)] += 0.25 * h_inv * _du_dz;
  }
}

/* ********************************** *
 *    wrappers on host and device     *
 * ********************************** */

/* Launch backward derivative computation */
void backward_derivative_device(float *dev_u, float *dev_du,
                                const int direction,
                                const GridSpec grid_spec,
                                const bool increment)
{
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz = grid_spec.nz;
  size_t nvoxels = grid_spec.number_of_voxels();
  if (not increment)
    CUDA_CHECK(cudaMemset(dev_du, 0, nvoxels * sizeof(float)));
  dim3 grid((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
            (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
            (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
  dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  if (direction == 0)
    backward_derivative_x_kernel<<<grid, block>>>(dev_u, dev_du, grid_spec);
  else if (direction == 1)
    backward_derivative_y_kernel<<<grid, block>>>(dev_u, dev_du, grid_spec);
  else if (direction == 2)
    backward_derivative_z_kernel<<<grid, block>>>(dev_u, dev_du, grid_spec);
  else
    throw std::runtime_error("Invalid direction");
}

/* Compute backward derivative on host */
void backward_derivative_host(float *u, float *du,
                              const int direction,
                              const GridSpec grid_spec,
                              const bool increment)
{
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz = grid_spec.nz;
  size_t nvoxels = grid_spec.number_of_voxels();
  if (not(increment))
    std::fill(du, du + nvoxels, 0);
  if (direction == 0)
  {
    float h_inv = grid_spec.nx / grid_spec.Lx;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
          float _du_dx = 0;
          _du_dx += u[IDX(nx, ny, nz, i, j, k)];
          _du_dx += u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, k)];
          _du_dx += u[IDX(nx, ny, nz, i, j, (k - 1 + nz) % nz)];
          _du_dx += u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, (k - 1 + nz) % nz)];
          _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, k)];
          _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny, k)];
          _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, (k - 1 + nz) % nz)];
          _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny,
                          (k - 1 + nz) % nz)];
          du[IDX(nx, ny, nz, i, j, k)] += 0.25 * h_inv * _du_dx;
        }
  }
  else if (direction == 1)
  {
    float h_inv = grid_spec.ny / grid_spec.Ly;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
          float _du_dy = 0;
          _du_dy += u[IDX(nx, ny, nz, i, j, k)];
          _du_dy += u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, k)];
          _du_dy += u[IDX(nx, ny, nz, i, j, (k - 1 + nz) % nz)];
          _du_dy += u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, (k - 1 + nz) % nz)];
          _du_dy -= u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, k)];
          _du_dy -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny, k)];
          _du_dy -= u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, (k - 1 + nz) % nz)];
          _du_dy -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny, (k - 1 + nz) % nz)];
          du[IDX(nx, ny, nz, i, j, k)] += 0.25 * h_inv * _du_dy;
        }
  }
  else if (direction == 2)
  {
    float h_inv = grid_spec.nz / grid_spec.Lz;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
          float _du_dz = 0;
          _du_dz += u[IDX(nx, ny, nz, i, j, k)];
          _du_dz += u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, k)];
          _du_dz += u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, k)];
          _du_dz += u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny, k)];
          _du_dz -= u[IDX(nx, ny, nz, i, j, (k - 1 + nz) % nz)];
          _du_dz -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, (k - 1 + nz) % nz)];
          _du_dz -= u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, (k - 1 + nz) % nz)];
          _du_dz -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny, (k - 1 + nz) % nz)];
          du[IDX(nx, ny, nz, i, j, k)] += 0.25 * h_inv * _du_dz;
        }
  }
  else
    throw std::runtime_error("Invalid direction");
}

/* Compute backward divergence of symmetric tensor on device */
void backward_divergence_device(float *dev_sigma, float *dev_div_sigma,
                                const GridSpec grid_spec)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  // Derivatives in x-direction
  backward_derivative_device(dev_sigma + 0 * nvoxels, dev_div_sigma + 0 * nvoxels, 0, grid_spec, false);
  backward_derivative_device(dev_sigma + 3 * nvoxels, dev_div_sigma + 1 * nvoxels, 0, grid_spec, false);
  backward_derivative_device(dev_sigma + 4 * nvoxels, dev_div_sigma + 2 * nvoxels, 0, grid_spec, false);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Derivatives in y-direction
  backward_derivative_device(dev_sigma + 3 * nvoxels, dev_div_sigma + 0 * nvoxels, 1, grid_spec, true);
  backward_derivative_device(dev_sigma + 1 * nvoxels, dev_div_sigma + 1 * nvoxels, 1, grid_spec, true);
  backward_derivative_device(dev_sigma + 5 * nvoxels, dev_div_sigma + 2 * nvoxels, 1, grid_spec, true);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Derivatives in z-direction
  backward_derivative_device(dev_sigma + 4 * nvoxels, dev_div_sigma + 0 * nvoxels, 2, grid_spec, true);
  backward_derivative_device(dev_sigma + 5 * nvoxels, dev_div_sigma + 1 * nvoxels, 2, grid_spec, true);
  backward_derivative_device(dev_sigma + 2 * nvoxels, dev_div_sigma + 2 * nvoxels, 2, grid_spec, true);
  CUDA_CHECK(cudaDeviceSynchronize());
}

/* Compute backward divergence of symmetric tensor on host */
void backward_divergence_host(float *sigma, float *div_sigma,
                              const GridSpec grid_spec)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  // Derivatives in x-direction
  backward_derivative_host(sigma + 0 * nvoxels, div_sigma + 0 * nvoxels, 0, grid_spec, false);
  backward_derivative_host(sigma + 3 * nvoxels, div_sigma + 1 * nvoxels, 0, grid_spec, false);
  backward_derivative_host(sigma + 4 * nvoxels, div_sigma + 2 * nvoxels, 0, grid_spec, false);
  // Derivatives in y-direction
  backward_derivative_host(sigma + 3 * nvoxels, div_sigma + 0 * nvoxels, 1, grid_spec, true);
  backward_derivative_host(sigma + 1 * nvoxels, div_sigma + 1 * nvoxels, 1, grid_spec, true);
  backward_derivative_host(sigma + 5 * nvoxels, div_sigma + 2 * nvoxels, 1, grid_spec, true);
  // Derivatives in z-direction
  backward_derivative_host(sigma + 4 * nvoxels, div_sigma + 0 * nvoxels, 2, grid_spec, true);
  backward_derivative_host(sigma + 5 * nvoxels, div_sigma + 1 * nvoxels, 2, grid_spec, true);
  backward_derivative_host(sigma + 2 * nvoxels, div_sigma + 2 * nvoxels, 2, grid_spec, true);
}