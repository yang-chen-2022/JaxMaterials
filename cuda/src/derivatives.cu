#include "derivatives.hh"
/** @brief implementation of derivatives.hh */

/* ********************************** *
 *    naive CUDA kernels              *
 * ********************************** */

/* Kernel for backward derivative computation in x-direction */
__global__ void backward_derivative_x_kernel(float *u, float *du_dx,
                                             const GridSpec grid_spec)
{
  float h_inv = grid_spec.nx / grid_spec.Lx;
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
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
    du_dx[IDX(nx, ny, nz, a, b, c)] = 0.25 * h_inv * _du_dx;
  }
}

/* Kernel for backward derivative computation in y-direction */
__global__ void backward_derivative_y_kernel(float *u, float *du_dy,
                                             const GridSpec grid_spec)
{
  float h_inv = grid_spec.ny / grid_spec.Ly;
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
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
    du_dy[IDX(nx, ny, nz, a, b, c)] = 0.25 * h_inv * _du_dy;
  }
}

/* Kernel for backward derivative computation in y-direction */
__global__ void backward_derivative_z_kernel(float *u, float *du_dz,
                                             const GridSpec grid_spec)
{
  float h_inv = grid_spec.nz / grid_spec.Lz;
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
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
    du_dz[IDX(nx, ny, nz, a, b, c)] = 0.25 * h_inv * _du_dz;
  }
}

/* ********************************** *
 *    shared memory CUDA kernels      *
 * ********************************** */

/* Copy data into shared memory patch which allows accessing u_{a,b,c} with
 * -1 <= a < nx, -1 <= b < ny, -1 <= c < nz. This allows the computation of
 * all backward derivatives.
 */
__device__ void copy_patch_into_shared_memory(float *u, float *u_shared,
                                              const GridSpec grid_spec)
{
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;
  if ((a < nx) && (b < ny) && (c < nz))
  {
    // Copy into shared memory
    u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x + 1,
                 threadIdx.y + 1, threadIdx.z + 1)] = u[IDX(nx, ny, nz, a, b, c)];
    if (threadIdx.x == 0)
      u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x,
                   threadIdx.y + 1, threadIdx.z + 1)] =
          u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, c)];
    if (threadIdx.y == 0)
      u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x + 1,
                   threadIdx.y, threadIdx.z + 1)] =
          u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, c)];
    if (threadIdx.z == 0)
      u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x + 1,
                   threadIdx.y + 1, threadIdx.z)] =
          u[IDX(nx, ny, nz, a, b, (c - 1 + nz) % nz)];
    if ((threadIdx.x == 0) && (threadIdx.y == 0))
      u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x,
                   threadIdx.y, threadIdx.z + 1)] =
          u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny, c)];
    if ((threadIdx.y == 0) && (threadIdx.z == 0))
      u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x + 1,
                   threadIdx.y, threadIdx.z)] =
          u[IDX(nx, ny, nz, a, (b - 1 + ny) % ny, (c - 1 + nz) % nz)];
    if ((threadIdx.z == 0) && (threadIdx.x == 0))
      u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x,
                   threadIdx.y + 1, threadIdx.z)] =
          u[IDX(nx, ny, nz, (a - 1 + nx) % nx, b, (c - 1 + nz) % nz)];
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
      u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1, threadIdx.x,
                   threadIdx.y, threadIdx.z)] =
          u[IDX(nx, ny, nz, (a - 1 + nx) % nx, (b - 1 + ny) % ny,
                (c - 1 + nz) % nz)];
  }
}

/* Kernel for backward derivative computation in x-direction */
__global__ void backward_derivative_x_kernel_shared(float *u, float *du_dx,
                                                    const GridSpec grid_spec)
{
  float h_inv = grid_spec.nx / grid_spec.Lx;
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  __shared__ float
      u_shared[(BLOCKSIZE_X + 1) * (BLOCKSIZE_Y + 1) * (BLOCKSIZE_Z + 1)];
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;
  copy_patch_into_shared_memory(u, u_shared, grid_spec);
  __syncthreads();
  if ((a < nx) && (b < ny) && (c < nz))
  {
    float _du_dx = 0;
    // apply derivative
    _du_dx += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)];
    _du_dx += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)];
    _du_dx += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)];
    _du_dx += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y, threadIdx.z)];
    _du_dx -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)];
    _du_dx -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y, threadIdx.z + 1)];
    _du_dx -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y + 1, threadIdx.z)];
    _du_dx -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y, threadIdx.z)];
    //  Copy back to global memory
    du_dx[IDX(nx, ny, nz, a, b, c)] = 0.25 * h_inv * _du_dx;
  }
}

/* Kernel for backward derivative computation in y-direction */
__global__ void backward_derivative_y_kernel_shared(float *u, float *du_dy,
                                                    const GridSpec grid_spec)
{
  float h_inv = grid_spec.ny / grid_spec.Ly;
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  __shared__ float
      u_shared[(BLOCKSIZE_X + 1) * (BLOCKSIZE_Y + 1) * (BLOCKSIZE_Z + 1)];
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;
  copy_patch_into_shared_memory(u, u_shared, grid_spec);
  __syncthreads();
  if ((a < nx) && (b < ny) && (c < nz))
  {
    float _du_dy = 0;
    // apply derivative
    _du_dy += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)];
    _du_dy += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)];
    _du_dy += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)];
    _du_dy += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y + 1, threadIdx.z)];
    _du_dy -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)];
    _du_dy -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y, threadIdx.z + 1)];
    _du_dy -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y, threadIdx.z)];
    _du_dy -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y, threadIdx.z)];
    //  Copy back to global memory
    du_dy[IDX(nx, ny, nz, a, b, c)] = 0.25 * h_inv * _du_dy;
  }
}

/* Kernel for backward derivative computation in y-direction */
__global__ void backward_derivative_z_kernel_shared(float *u, float *du_dz,
                                                    const GridSpec grid_spec)
{
  float h_inv = grid_spec.nz / grid_spec.Lz;
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  __shared__ float
      u_shared[(BLOCKSIZE_X + 1) * (BLOCKSIZE_Y + 1) * (BLOCKSIZE_Z + 1)];
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;
  copy_patch_into_shared_memory(u, u_shared, grid_spec);
  __syncthreads();
  if ((a < nx) && (b < ny) && (c < nz))
  {
    float _du_dz = 0;
    // apply derivative
    _du_dz += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1)];
    _du_dz += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y + 1, threadIdx.z + 1)];
    _du_dz += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y, threadIdx.z + 1)];
    _du_dz += u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y, threadIdx.z + 1)];
    _du_dz -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y + 1, threadIdx.z)];
    _du_dz -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y + 1, threadIdx.z)];
    _du_dz -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x + 1, threadIdx.y, threadIdx.z)];
    _du_dz -= u_shared[IDX(BLOCKSIZE_X + 1, BLOCKSIZE_Y + 1, BLOCKSIZE_Z + 1,
                           threadIdx.x, threadIdx.y, threadIdx.z)];
    //  Copy back to global memory
    du_dz[IDX(nx, ny, nz, a, b, c)] = 0.25 * h_inv * _du_dz;
  }
}

/* ********************************** *
 *    wrappers on host and device     *
 * ********************************** */

/* Launch backward derivative computation */
void backward_derivative_device(float *dev_u, float *dev_du,
                                const int direction,
                                const GridSpec grid_spec,
                                bool use_shared_memory)
{
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  dim3 grid((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
            (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
            (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
  dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  if (use_shared_memory)
  {
    if (direction == 0)
      backward_derivative_x_kernel_shared<<<grid, block>>>(dev_u, dev_du, grid_spec);
    else if (direction == 1)
      backward_derivative_y_kernel_shared<<<grid, block>>>(dev_u, dev_du, grid_spec);
    else if (direction == 2)
      backward_derivative_z_kernel_shared<<<grid, block>>>(dev_u, dev_du, grid_spec);
    else
      throw std::runtime_error("Invalid direction");
  }
  else
  {
    if (direction == 0)
      backward_derivative_x_kernel<<<grid, block>>>(dev_u, dev_du, grid_spec);
    else if (direction == 1)
      backward_derivative_y_kernel<<<grid, block>>>(dev_u, dev_du, grid_spec);
    else if (direction == 2)
      backward_derivative_z_kernel<<<grid, block>>>(dev_u, dev_du, grid_spec);
    else
      throw std::runtime_error("Invalid direction");
  }
}

/* Compute backward derivative on host */
void backward_derivative_host(float *u, float *du,
                              const int direction,
                              const GridSpec grid_spec)
{
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
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
          du[IDX(nx, ny, nz, i, j, k)] = 0.25 * h_inv * _du_dx;
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
          du[IDX(nx, ny, nz, i, j, k)] = 0.25 * h_inv * _du_dy;
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
          du[IDX(nx, ny, nz, i, j, k)] = 0.25 * h_inv * _du_dz;
        }
  }
  else
    throw std::runtime_error("Invalid direction");
}
