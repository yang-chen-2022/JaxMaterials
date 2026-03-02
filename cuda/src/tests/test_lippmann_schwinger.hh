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
  float sigma_avg[6];
  for (int alpha = 0; alpha < 6; ++alpha)
  {
    sigma_avg[alpha] = 0;
    for (int j = 0; j < ncells; ++j)
      sigma_avg[alpha] += sigma[alpha * ncells + j];
    sigma_avg[alpha] /= ncells;
  }

  float reldiv_real = vector_norm(div_sigma, ncells) / tensor_norm(sigma_avg, 1);
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
  float rel_diff = (reldiv_fourier - reldiv_real) / reldiv_real;
  // free memory
  CUDA_CHECK(cudaFree(dev_sigma));
  CUDA_CHECK(cudaFree(dev_sigma_hat));
  CUDA_CHECK(cudaFreeHost(sigma));
  CUDA_CHECK(cudaFreeHost(div_sigma));
  float tolerance = 1.E-4;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/** @brief Check whether solver converges in zero iterations for homogeneous materials
 */
TEST_F(LippmannSchwingerTest, TestHomogeneousMaterial)
{
  int ncells = grid_spec.number_of_cells();
  float *mu = nullptr;
  float *lambda = nullptr;
  float *epsilon = nullptr;
  float *sigma = nullptr;
  float epsilon_bar[6] = {1.0, 0.4, 0.3, 0.1, -0.4, 0.7};
  CUDA_CHECK(cudaMallocHost(&mu, ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&lambda, ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * ncells * sizeof(float)));
  std::fill(mu, mu + ncells, 1.2);
  std::fill(lambda, lambda + ncells, 1.2);

  LippmannSchwingerSolver solver(grid_spec);
  int iter = solver.apply(lambda, mu, epsilon_bar, epsilon, sigma, 1.E-4, 32);
  CUDA_CHECK(cudaFreeHost(mu));
  CUDA_CHECK(cudaFreeHost(lambda));
  CUDA_CHECK(cudaFreeHost(epsilon));
  CUDA_CHECK(cudaFreeHost(sigma));
  EXPECT_EQ(iter, 0);
}

#endif // TEST_LIPPMANN_SCHWINGER_HH