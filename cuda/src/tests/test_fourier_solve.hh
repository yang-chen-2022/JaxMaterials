#ifndef TEST_FOURIER_SOLVE_HH
#define TEST_FOURIER_SOLVE_HH TEST_FOURIER_SOLVE_HH
#include <random>
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
  float *tau_hat = nullptr;
  float *dev_tau_hat = nullptr;
  float *dev_epsilon_hat = nullptr;
  float *epsilon_hat = nullptr;
  CUDA_CHECK(cudaMallocHost(&xi_zero, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&tau_hat, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon_hat, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_tau_hat, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_epsilon_hat, 6 * ncells * sizeof(float)));

  // Initialise Fourier vectors
  initialize_xizero_host(xi_zero, grid_spec);
  initialize_xizero(dev_xi_zero, grid_spec);

  // Initialize with random numbers
  std::generate(tau_hat, tau_hat + 6 * ncells, [&]()
                { return distribution(rng); });
  CUDA_CHECK(cudaMemcpy(dev_tau_hat, tau_hat, 6 * ncells * sizeof(float), cudaMemcpyDefault));

  // Solve on device and copy back
  fourier_solve(dev_tau_hat, dev_epsilon_hat, dev_xi_zero,
                lambda_0, mu_0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(epsilon_hat, dev_epsilon_hat, 6 * ncells * sizeof(float), cudaMemcpyDefault));
  float sigma_nrm2 = 0;
  float div_nrm2 = 0;
  for (int ell = 0; ell < ncells; ++ell)
  {
    float xi[3];
    float xi_sigma[3];
    float tau[6];
    float epsilon[6];
    float sigma[6];
    for (int mu = 0; mu < 3; ++mu)
      xi[mu] = xi_zero[mu * ncells + ell];

    for (int mu = 0; mu < 6; ++mu)
    {
      tau[mu] = tau_hat[mu * ncells + ell];
      epsilon[mu] = epsilon_hat[mu * ncells + ell];
    }

    float tr_epsilon = epsilon[0] + epsilon[1] + epsilon[2];

    sigma[0] = tau[0] + 2 * mu_0 * epsilon[0] + lambda_0 * tr_epsilon;
    sigma[1] = tau[1] + 2 * mu_0 * epsilon[1] + lambda_0 * tr_epsilon;
    sigma[2] = tau[2] + 2 * mu_0 * epsilon[2] + lambda_0 * tr_epsilon;
    sigma[3] = tau[3] + 2 * mu_0 * epsilon[3];
    sigma[4] = tau[4] + 2 * mu_0 * epsilon[4];
    sigma[5] = tau[5] + 2 * mu_0 * epsilon[5];

    xi_sigma[0] = xi[0] * sigma[0] + xi[1] * sigma[5] + xi[2] * sigma[4];
    xi_sigma[1] = xi[0] * sigma[5] + xi[1] * sigma[1] + xi[2] * sigma[3];
    xi_sigma[2] = xi[0] * sigma[4] + xi[1] * sigma[3] + xi[2] * sigma[2];
    for (int mu = 0; mu < 3; ++mu)
      div_nrm2 += xi_sigma[mu] * xi_sigma[mu];
    for (int mu = 0; mu < 6; ++mu)
    {
      sigma_nrm2 += sigma[mu] * sigma[mu];
    }
  }
  float rel_diff = sqrt(div_nrm2 / sigma_nrm2);
  // free memory
  CUDA_CHECK(cudaFree(dev_xi_zero));
  CUDA_CHECK(cudaFree(dev_tau_hat));
  CUDA_CHECK(cudaFree(dev_epsilon_hat));
  CUDA_CHECK(cudaFreeHost(xi_zero));
  CUDA_CHECK(cudaFreeHost(tau_hat));
  CUDA_CHECK(cudaFreeHost(epsilon_hat));
  float tolerance = 1.E-6;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

#endif // TEST_FOURIER_SOLVE_HH