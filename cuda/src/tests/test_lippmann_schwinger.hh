#ifndef TEST_LIPPMANN_SCHWINGER_HH
#define TEST_LIPPMANN_SCHWINGER_HH TEST_LIPPMANN_SCHWINGER_HH
#include <random>
#include "cufft.h"
#include <algorithm>
#include "common.hh"
#include "lippmann_schwinger.hh"
#include <gtest/gtest.h>

class LippmannSchwingerTest : public ::testing::Test
{
public:
  /** @Create a new instance */
  LippmannSchwingerTest() {}

protected:
  /** @brief initialise tests */
  void SetUp() override
  {
    grid_spec.nx = 48;
    grid_spec.ny = 64;
    grid_spec.nz = 32;
    grid_spec.Lx = 1.1;
    grid_spec.Ly = 0.9;
    grid_spec.Lz = 0.7;
  }
  void TearDown() override
  {
  }
  /* Grid specification */
  GridSpec grid_spec;
};

/** @brief Check whether relative divergence is computed consistently
 */
TEST_F(LippmannSchwingerTest, TestRelativeDivergence)
{
  /* Random number generator */
  std::default_random_engine rng;
  /* normal distribution */
  std::normal_distribution<float> distribution;

  int ncells = grid_spec.number_of_cells();
  cufftComplex *dev_sigma_hat = nullptr;
  cufftComplex *dev_sigma = nullptr;
  float *sigma = nullptr;
  float *div_sigma = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_sigma, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_sigma_hat, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&div_sigma, 3 * ncells * sizeof(float)));

  // Initialize sigma with random numbers
  std::generate(sigma, sigma + 6 * ncells, [&]()
                { return distribution(rng); });

  backward_divergence_host(sigma, div_sigma, grid_spec);
  float reldiv_real = vector_norm(div_sigma, ncells) / tensor_norm(sigma, ncells);
  LippmannSchwingerSolver solver(grid_spec);

  /* cuFFT plan */
  cufftHandle plan;
  int n[3] = {grid_spec.nz, grid_spec.ny, grid_spec.nx};
  CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, ncells, n, 1, ncells, CUFFT_C2C, 6));
  CUDA_CHECK(cudaMemset(dev_sigma, 0, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMemcpy2D(dev_sigma, 2 * sizeof(float), sigma, sizeof(float), sizeof(float), 6 * ncells, cudaMemcpyHostToDevice));
  // Fourier transform sigma
  CUFFT_CHECK(cufftExecC2C(plan, dev_sigma, dev_sigma_hat, CUFFT_FORWARD));
  CUDA_CHECK(cudaDeviceSynchronize());

  float reldiv_fourier = solver.relative_divergence_norm(dev_sigma_hat);
  printf("%8.4e %8.4e\n", reldiv_real, reldiv_fourier);
  // free memory
  CUDA_CHECK(cudaFree(dev_sigma));
  CUDA_CHECK(cudaFree(dev_sigma_hat));
  CUDA_CHECK(cudaFreeHost(sigma));
  CUDA_CHECK(cudaFreeHost(div_sigma));
}

#endif // TEST_LIPPMANN_SCHWINGER_HH