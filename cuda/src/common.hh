/** @brief Some common functionality */
#ifndef COMMON_HH
#define COMMON_HH COMMON_HH

#include <math.h>
#include <stdio.h>
#include "cufft.h"

// Block size used for 3d kernel launches
#define BLOCKSIZE_X 8
#define BLOCKSIZE_Y 8
#define BLOCKSIZE_Z 8

// Block size used for 1d kernel launches
#define BLOCKSIZE 1024

// maximal domain size in any direction
#define NMAX 4096

// Convert 3d index to linear index
#define IDX(Nx, Ny, Nz, i, j, k) ((i) + (Nx) * ((j) + (Ny) * (k)))
// Convert (1+3)d index to linear index
#define FIDX(Nx, Ny, Nz, mu, i, j, k) \
  ((i) + (Nx) * ((j) + (Ny) * ((k) + (Nz) * (mu))))

// Error checking, as defined in CUDA Programming guide, Section 2.1.7
#define CUDA_CHECK(expr_to_check)                                      \
  do                                                                   \
  {                                                                    \
    cudaError_t result = expr_to_check;                                \
    if (result != cudaSuccess)                                         \
    {                                                                  \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, \
              __LINE__, result, cudaGetErrorString(result));           \
    }                                                                  \
  } while (0)

// Corresponding cuFFT error checking
#define CUFFT_CHECK(expr_to_check)                                           \
  do                                                                         \
  {                                                                          \
    cufftResult_t result = expr_to_check;                                    \
    if (result != CUFFT_SUCCESS)                                             \
    {                                                                        \
      fprintf(stderr, "cuFFT Runtime Error: %s:%i:%d\n", __FILE__, __LINE__, \
              result);                                                       \
    }                                                                        \
  } while (0)

// Corresponding cuBLAS error checking
#define CUBLAS_CHECK(expr_to_check)                                           \
  do                                                                          \
  {                                                                           \
    cublasStatus_t stat = expr_to_check;                                      \
    if (stat != CUBLAS_STATUS_SUCCESS)                                        \
    {                                                                         \
      fprintf(stderr, "cuBLAS Runtime Error: %s:%i:%d\n", __FILE__, __LINE__, \
              stat);                                                          \
    }                                                                         \
  } while (0)

/** @brief Specification of computational grid
 *
 * Describes grid of the domain Lx x Ly x Lz with nx, ny, nz grid cells
 * (or voxels) in the different coordinate directions
 */
struct GridSpec
{
  int nx;   // Number of grid cells in x-direction
  int ny;   // Number of grid cells in y-direction
  int nz;   // Number of grid cells in z-direction
  float Lx; // Size of domain in x-direction
  float Ly; // Size of domain in y-direction
  float Lz; // Size of domain in z-direction
  /** @brief Return total number of grid cells */
  int number_of_cells() const { return nx * ny * nz; }
};

/** @brief Compute relative L2 norm
 *
 * Computes ||u-u_{ref}||_2 / ||u_{ref}||_2 which can be used for testing and
 * debugging
 *
 * @param[out] u: field (host pointer)
 * @param[out] u_ref: reference field to compare to (host pointer)
 * @param[in] ndof number of unknowns
 */
float relative_difference(float *u, float *u_ref, const int ndof);

/* @brief Compute norm of vector field */
float vector_norm(float *u, const int ncells);
float vector_norm(cufftComplex *u, const int ncells);

/* @brief Compute norm of tensor field */
float tensor_norm(float *u, const int ncells);
float tensor_norm(cufftComplex *u, const int ncells);

#endif // COMMON_HH
