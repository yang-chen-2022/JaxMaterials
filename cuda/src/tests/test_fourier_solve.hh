#ifndef TEST_FOURIER_SOLVE_HH
#define TEST_FOURIER_SOLVE_HH TEST_FOURIER_SOLVE_HH
#include <random>
#include "cufft.h"
#include <algorithm>
#include "fourier_solve.hh"
#include <gtest/gtest.h>

class FourierSolveTest : public ::testing::Test
{
public:
  /** @Create a new instance */
  FourierSolveTest() {}

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
  GridSpec grid_spec;
};

/** @brief Check whether xi-zero is constructed consistently on device and host
 */
TEST_F(FourierSolveTest, TestXiZero)
{
  float tolerance = 1.E-6;
  // halo size
  int ncells = grid_spec.number_of_cells();
  // allocate host memory
  float *xi_zero = nullptr;
  float *xi_zero_ref = nullptr;
  CUDA_CHECK(cudaMallocHost(&xi_zero, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&xi_zero_ref, 3 * ncells * sizeof(float)));

  // allocate device memory
  float *dev_xi_zero = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * ncells * sizeof(float)));
  initialize_xizero_host(xi_zero_ref, grid_spec);
  initialize_xizero(dev_xi_zero, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(xi_zero, dev_xi_zero, 3 * ncells * sizeof(float),
                        cudaMemcpyDefault));

  float rel_diff = relative_difference(xi_zero, xi_zero_ref, 3 * ncells);

  // Free memory
  CUDA_CHECK(cudaFreeHost(xi_zero));
  CUDA_CHECK(cudaFreeHost(xi_zero_ref));
  CUDA_CHECK(cudaFree(dev_xi_zero));

  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/* Check whether div(sigma^0) = 0 for homogeneous material
 *
 * Here sigma^0_{ij} = C^0_{ijkl} epsilon_{kl} + tau_{ij}is obtained by solving the
 * equations of linear elasticity in a homogeneous isotropic material with
 *
 *     C^0_{ijkl} = lambda^0 (delta_{ij}delta_{kl}
 *                + mu^0 (delta_{ik}delta_{jl} + delta_{il}delta_{jk}))
 *
 * and given, random tau
 */
TEST_F(FourierSolveTest, TestDivSigma)
{
  std::default_random_engine rng(7812481);
  std::normal_distribution<float> distribution(0, 1);
  int ncells = grid_spec.number_of_cells();
  const float lambda_0 = 0.8;
  const float mu_0 = 1.3;
  // allocate memory
  float *xi_zero = nullptr;
  float *dev_xi_zero = nullptr;
  float *tau = nullptr;
  cufftComplex *dev_tau = nullptr;
  cufftComplex *dev_tau_hat = nullptr;
  cufftComplex *tau_hat = nullptr;
  cufftComplex *dev_epsilon_hat = nullptr;
  cufftComplex *epsilon_hat = nullptr;
  CUDA_CHECK(cudaMallocHost(&xi_zero, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&tau, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&tau_hat, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&epsilon_hat, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_tau, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_tau_hat, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_epsilon_hat, 6 * ncells * sizeof(cufftComplex)));

  // Initialise Fourier vectors
  initialize_xizero_host(xi_zero, grid_spec);
  initialize_xizero(dev_xi_zero, grid_spec);

  // Initialize with random numbers
  std::generate(tau, tau + 6 * ncells, [&]()
                { return distribution(rng); });
  CUDA_CHECK(cudaMemset(dev_tau, 0, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMemcpy2D(dev_tau, 2 * sizeof(float), tau, sizeof(float), sizeof(float), 6 * ncells, cudaMemcpyDefault));

  cufftHandle plan;
  int n[3] = {grid_spec.nz, grid_spec.ny, grid_spec.nx};
  CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, ncells, n, 1, ncells, CUFFT_C2C, 6));
  CUFFT_CHECK(cufftExecC2C(plan, dev_tau, dev_tau_hat, CUFFT_FORWARD));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Solve on device and copy back
  fourier_solve_device(dev_tau_hat, dev_epsilon_hat, dev_xi_zero,
                       lambda_0, mu_0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(epsilon_hat, dev_epsilon_hat, 6 * ncells * sizeof(cufftComplex), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(tau_hat, dev_tau_hat, 6 * ncells * sizeof(cufftComplex), cudaMemcpyDefault));
  float sigma_nrm2 = 0;
  float div_nrm2 = 0;
  for (int ell = 0; ell < ncells; ++ell)
  {
    float xi[3];
    cufftComplex xi_sigma[3];
    cufftComplex tau[6];
    cufftComplex epsilon[6];
    cufftComplex sigma[6];
    for (int mu = 0; mu < 3; ++mu)
      xi[mu] = xi_zero[mu * ncells + ell];

    for (int mu = 0; mu < 6; ++mu)
    {
      tau[mu] = tau_hat[mu * ncells + ell];
      epsilon[mu] = epsilon_hat[mu * ncells + ell];
    }

    cufftComplex tr_epsilon;
    tr_epsilon.x = epsilon[0].x + epsilon[1].x + epsilon[2].x;
    tr_epsilon.y = epsilon[0].y + epsilon[1].y + epsilon[2].y;

    for (int alpha = 0; alpha < 6; ++alpha)
    {
      sigma[alpha].x = tau[alpha].x + 2 * mu_0 * epsilon[alpha].x;
      sigma[alpha].y = tau[alpha].y + 2 * mu_0 * epsilon[alpha].y;
      if (alpha < 3)
      {
        sigma[alpha].x += lambda_0 * tr_epsilon.x;
        sigma[alpha].y += lambda_0 * tr_epsilon.y;
      }
    }

    xi_sigma[0].x = xi[0] * sigma[0].x + xi[1] * sigma[5].x + xi[2] * sigma[4].x;
    xi_sigma[0].y = xi[0] * sigma[0].y + xi[1] * sigma[5].y + xi[2] * sigma[4].y;
    xi_sigma[1].x = xi[0] * sigma[5].x + xi[1] * sigma[1].x + xi[2] * sigma[3].x;
    xi_sigma[1].y = xi[0] * sigma[5].y + xi[1] * sigma[1].y + xi[2] * sigma[3].y;
    xi_sigma[2].x = xi[0] * sigma[4].x + xi[1] * sigma[3].x + xi[2] * sigma[2].x;
    xi_sigma[2].y = xi[0] * sigma[4].y + xi[1] * sigma[3].y + xi[2] * sigma[2].y;
    for (int mu = 0; mu < 3; ++mu)
      div_nrm2 += xi_sigma[mu].x * xi_sigma[mu].x + xi_sigma[mu].y * xi_sigma[mu].y;
    for (int mu = 0; mu < 6; ++mu)
    {
      sigma_nrm2 += sigma[mu].x * sigma[mu].x + sigma[mu].y * sigma[mu].y;
    }
  }
  float rel_diff = sqrt(div_nrm2 / sigma_nrm2);
  // free memory
  CUDA_CHECK(cudaFree(dev_xi_zero));
  CUDA_CHECK(cudaFree(dev_tau_hat));
  CUDA_CHECK(cudaFreeHost(tau_hat));
  CUDA_CHECK(cudaFree(dev_tau));
  CUDA_CHECK(cudaFree(dev_epsilon_hat));
  CUDA_CHECK(cudaFreeHost(xi_zero));
  CUDA_CHECK(cudaFreeHost(tau));
  CUDA_CHECK(cudaFreeHost(epsilon_hat));
  float tolerance = 1.E-6;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

#endif // TEST_FOURIER_SOLVE_HH