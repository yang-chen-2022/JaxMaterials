#ifndef TEST_FOURIER_SOLVE_HH
#define TEST_FOURIER_SOLVE_HH TEST_FOURIER_SOLVE_HH
#include <random>
#include "cufft.h"
#include <algorithm>
#include "derivatives.hh"
#include "fourier.hh"
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
                        cudaMemcpyDeviceToHost));

  float rel_diff = relative_difference(xi_zero, xi_zero_ref, 3 * ncells);

  // Free memory
  CUDA_CHECK(cudaFreeHost(xi_zero));
  CUDA_CHECK(cudaFreeHost(xi_zero_ref));
  CUDA_CHECK(cudaFree(dev_xi_zero));

  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/* Check whether div(sigma^0) = 0 for homogeneous material in Fourier space
 *
 * Here sigma^0_{ij} = C^0_{ijkl} epsilon_{kl} + tau_{ij} is obtained by solving the
 * equations of linear elasticity in a homogeneous isotropic material with
 *
 *     C^0_{ijkl} = lambda^0 (delta_{ij}delta_{kl}
 *                + mu^0 (delta_{ik}delta_{jl} + delta_{il}delta_{jk}))
 *
 * This test checks that div(sigma^0) = 0 in Fourier space.
 *
 * and given, random tau
 */
TEST_F(FourierSolveTest, TestDivSigmaFourier)
{
  // random number generator
  std::default_random_engine rng(7812481);
  std::normal_distribution<float> distribution(0, 1);
  int ncells = grid_spec.number_of_cells();
  // Lame parameters
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

  // Initialize tau with random numbers
  std::generate(tau, tau + 6 * ncells, [&]()
                { return distribution(rng); });
  CUDA_CHECK(cudaMemset(dev_tau, 0, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMemcpy2D(dev_tau, 2 * sizeof(float), tau, sizeof(float), sizeof(float), 6 * ncells, cudaMemcpyHostToDevice));

  // Fourier transform the real-valued tau
  cufftHandle plan;
  int n[3] = {grid_spec.nz, grid_spec.ny, grid_spec.nx};
  CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, ncells, n, 1, ncells, CUFFT_C2C, 6));
  CUFFT_CHECK(cufftExecC2C(plan, dev_tau, dev_tau_hat, CUFFT_FORWARD));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Solve on device and copy back
  fourier_solve_device(dev_tau_hat, dev_epsilon_hat, dev_xi_zero,
                       lambda_0, mu_0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(epsilon_hat, dev_epsilon_hat, 6 * ncells * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(tau_hat, dev_tau_hat, 6 * ncells * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  float sigma_nrm2 = 0;
  float div_nrm2 = 0;
  // Compute ||div(sigma^0)||^2 and ||sigma^0||^2 in Fourier space
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

    // trace of epsilon
    cufftComplex tr_epsilon;
    tr_epsilon.x = epsilon[0].x + epsilon[1].x + epsilon[2].x;
    tr_epsilon.y = epsilon[0].y + epsilon[1].y + epsilon[2].y;

    // compute sigma
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

    // Compute three components of dot-product xi.sigma
    xi_sigma[0].x = xi[0] * sigma[0].x + xi[1] * sigma[3].x + xi[2] * sigma[4].x;
    xi_sigma[0].y = xi[0] * sigma[0].y + xi[1] * sigma[3].y + xi[2] * sigma[4].y;
    xi_sigma[1].x = xi[0] * sigma[3].x + xi[1] * sigma[1].x + xi[2] * sigma[5].x;
    xi_sigma[1].y = xi[0] * sigma[3].y + xi[1] * sigma[1].y + xi[2] * sigma[5].y;
    xi_sigma[2].x = xi[0] * sigma[4].x + xi[1] * sigma[5].x + xi[2] * sigma[2].x;
    xi_sigma[2].y = xi[0] * sigma[4].y + xi[1] * sigma[5].y + xi[2] * sigma[2].y;
    // update (squared) norms
    for (int mu = 0; mu < 3; ++mu)
      div_nrm2 += xi_sigma[mu].x * xi_sigma[mu].x + xi_sigma[mu].y * xi_sigma[mu].y;
    for (int mu = 0; mu < 3; ++mu)
    {
      sigma_nrm2 += sigma[mu].x * sigma[mu].x + sigma[mu].y * sigma[mu].y;
    }
    for (int mu = 3; mu < 6; ++mu)
    {
      sigma_nrm2 += 2 * (sigma[mu].x * sigma[mu].x + sigma[mu].y * sigma[mu].y);
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

/* Check whether div(sigma^0) = 0 for homogeneous material in real space
 *
 * Here sigma^0_{ij} = C^0_{ijkl} epsilon_{kl} + tau_{ij} is obtained by solving the
 * equations of linear elasticity in a homogeneous isotropic material with
 *
 *     C^0_{ijkl} = lambda^0 (delta_{ij}delta_{kl}
 *                + mu^0 (delta_{ik}delta_{jl} + delta_{il}delta_{jk}))
 *
 * This test checks that div(sigma^0) = 0 in real space.
 *
 * and given, random tau
 */
TEST_F(FourierSolveTest, TestDivSigma)
{
  // random number generator
  std::default_random_engine rng(7812481);
  std::normal_distribution<float> distribution(0, 1);
  int ncells = grid_spec.number_of_cells();
  // Lame parameters
  const float lambda_0 = 0.8;
  const float mu_0 = 1.3;
  // allocate memory
  float *dev_xi_zero = nullptr;
  float *tau = nullptr;
  cufftComplex *dev_tau = nullptr;
  cufftComplex *dev_tau_hat = nullptr;
  cufftComplex *dev_epsilon_hat = nullptr;
  cufftComplex *dev_epsilon = nullptr;
  float *epsilon = nullptr;
  float *sigma = nullptr;
  float *div_sigma = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&tau, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_tau, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_tau_hat, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_epsilon, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_epsilon_hat, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&epsilon, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&div_sigma, 3 * ncells * sizeof(float)));

  // Initialise Fourier vectors
  initialize_xizero(dev_xi_zero, grid_spec);

  // Initialize tau with random numbers
  std::generate(tau, tau + 6 * ncells, [&]()
                { return distribution(rng); });
  CUDA_CHECK(cudaMemset(dev_tau, 0, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMemcpy2D(dev_tau, 2 * sizeof(float), tau, sizeof(float), sizeof(float), 6 * ncells, cudaMemcpyHostToDevice));

  // Fourier transform the real-valued tau
  cufftHandle plan;
  int n[3] = {grid_spec.nz, grid_spec.ny, grid_spec.nx};
  CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, ncells, n, 1, ncells, CUFFT_C2C, 6));
  CUFFT_CHECK(cufftExecC2C(plan, dev_tau, dev_tau_hat, CUFFT_FORWARD));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Solve on device and Fourier transform back
  fourier_solve_device(dev_tau_hat, dev_epsilon_hat, dev_xi_zero,
                       lambda_0, mu_0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUFFT_CHECK(cufftExecC2C(plan, dev_epsilon_hat, dev_epsilon, CUFFT_INVERSE));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back to host
  CUDA_CHECK(cudaMemcpy2D(epsilon, sizeof(float), dev_epsilon, 2 * sizeof(float), sizeof(float), 6 * ncells, cudaMemcpyDeviceToHost));

  // Scale by 1/ncells to account for the fact that the inverse Fourier transform is not
  // normalised in cuFFT
  for (int ell = 0; ell < 6 * ncells; ++ell)
    epsilon[ell] *= 1.0 / ncells;
  // Compute stress sigma_{ij} = C^0_{ijkl} epsilon_{kl} + tau_{ij}
  for (int ell = 0; ell < ncells; ++ell)
  {
    float tr_epsilon = epsilon[0 * ncells + ell] + epsilon[1 * ncells + ell] + epsilon[2 * ncells + ell];
    for (int alpha = 0; alpha < 6; ++alpha)
    {
      int idx = alpha * ncells + ell;
      sigma[idx] = tau[idx] + 2 * mu_0 * epsilon[idx];
      if (alpha < 3)
      {
        sigma[idx] += lambda_0 * tr_epsilon;
      }
    }
  }
  // Compute divergence
  backward_divergence_host(sigma, div_sigma, grid_spec);

  float sigma_nrm2 = 0;
  float div_nrm2 = 0;
  for (int ell = 0; ell < ncells; ++ell)
  {
    for (int alpha = 0; alpha < 3; ++alpha)
      div_nrm2 += div_sigma[alpha * ncells + ell] * div_sigma[alpha * ncells + ell];
    for (int alpha = 0; alpha < 3; ++alpha)
    {
      sigma_nrm2 += sigma[alpha * ncells + ell] * sigma[alpha * ncells + ell];
    }
    for (int alpha = 3; alpha < 6; ++alpha)
    {
      sigma_nrm2 += 2 * sigma[alpha * ncells + ell] * sigma[alpha * ncells + ell];
    }
  }
  float rel_diff = sqrt(div_nrm2 / sigma_nrm2);
  // free memory
  CUDA_CHECK(cudaFree(dev_xi_zero));
  CUDA_CHECK(cudaFreeHost(tau));
  CUDA_CHECK(cudaFree(dev_tau));
  CUDA_CHECK(cudaFree(dev_tau_hat));
  CUDA_CHECK(cudaFree(dev_epsilon_hat));
  CUDA_CHECK(cudaFree(dev_epsilon));
  CUDA_CHECK(cudaFreeHost(epsilon));
  CUDA_CHECK(cudaFreeHost(sigma));
  CUDA_CHECK(cudaFreeHost(div_sigma));
  float tolerance = 5.E-5;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

#endif // TEST_FOURIER_SOLVE_HH