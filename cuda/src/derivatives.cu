#include "derivatives.hh"
/** @brief implementation of derivatives.hh */

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
__global__ void backward_derivative_x_kernel(float *u, float *du_dx,
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

/* Launch backward derivative computation in x-direction */
void backward_derivative_x(float *dev_u, float *dev_du_dx, const GridSpec grid_spec)
{
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  dim3 grid((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
            (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
            (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
  dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
  backward_derivative_x_kernel<<<grid, block>>>(dev_u, dev_du_dx, grid_spec);
}

/* Compute backward derivative in x-direction on host */
void backward_derivative_x_host(float *u, float *du_dx,
                                const GridSpec grid_spec)
{
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  float h_inv = grid_spec.nx / grid_spec.Lx;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
      {
        float _du_dx = 0;
        // apply derivative
        _du_dx += u[IDX(nx, ny, nz, i, j, k)];
        _du_dx += u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, k)];
        _du_dx += u[IDX(nx, ny, nz, i, j, (k - 1 + nz) % nz)];
        _du_dx += u[IDX(nx, ny, nz, i, (j - 1 + ny) % ny, (k - 1 + nz) % nz)];
        _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, k)];
        _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny, k)];
        _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, j, (k - 1 + nz) % nz)];
        _du_dx -= u[IDX(nx, ny, nz, (i - 1 + nx) % nx, (j - 1 + ny) % ny,
                        (k - 1 + nz) % nz)];
        du_dx[IDX(nx, ny, nz, i, j, k)] = 0.25 * h_inv * _du_dx;
      }
}
