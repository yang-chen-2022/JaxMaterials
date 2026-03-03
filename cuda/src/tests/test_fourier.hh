/** @brief Test Fourier methods */
#ifndef TEST_FOURIER_HH
#define TEST_FOURIER_HH TEST_FOURIER_HH
#include <random>
#include "cufft.h"
#include <algorithm>
#include "common.hh"
#include "derivatives.hh"
#include "fourier.hh"
#include <gtest/gtest.h>

class FourierTest : public ::testing::Test
{
public:
  /** @brief Constructor
   *
   * Create a new instance */
  FourierTest() {}

protected:
  /** @brief Initialise tests */
  void SetUp() override
  {
    grid_spec.nx = 48;
    grid_spec.ny = 64;
    grid_spec.nz = 32;
    grid_spec.Lx = 1.1;
    grid_spec.Ly = 0.9;
    grid_spec.Lz = 0.7;
    size_t nvoxels = grid_spec.number_of_voxels();

    // allocate memory
    CUDA_CHECK(cudaMallocHost(&xi_zero, 3 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&tau, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&tau_hat, 6 * nvoxels * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(&epsilon_hat, 6 * nvoxels * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(&epsilon, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&div_sigma, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_tau, 6 * nvoxels * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&dev_tau_hat, 6 * nvoxels * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&dev_epsilon_hat, 6 * nvoxels * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&dev_epsilon, 6 * nvoxels * sizeof(cufftComplex)));

    // Initialise Fourier vectors
    initialize_xizero_host(xi_zero, grid_spec);
    initialize_xizero_device(dev_xi_zero, grid_spec);
    int n[3] = {(int)grid_spec.nz, (int)grid_spec.ny, (int)grid_spec.nx};
    CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, nvoxels, n, 1, nvoxels, CUFFT_C2C, 6));
  }
  void TearDown() override
  {
    // free memory
    CUDA_CHECK(cudaFree(dev_xi_zero));
    CUDA_CHECK(cudaFree(dev_tau_hat));
    CUDA_CHECK(cudaFreeHost(tau_hat));
    CUDA_CHECK(cudaFree(dev_tau));
    CUDA_CHECK(cudaFree(dev_epsilon_hat));
    CUDA_CHECK(cudaFree(dev_epsilon));
    CUDA_CHECK(cudaFreeHost(xi_zero));
    CUDA_CHECK(cudaFreeHost(tau));
    CUDA_CHECK(cudaFreeHost(epsilon_hat));
    CUDA_CHECK(cudaFreeHost(epsilon));
    CUDA_CHECK(cudaFreeHost(sigma));
    CUDA_CHECK(cudaFreeHost(div_sigma));
  }
  /* Grid specification */
  GridSpec grid_spec;
  /* constant reference Lame parameters */
  const float lambda_0 = 0.8;
  const float mu_0 = 1.3;
  /* Fourier vector on host */
  float *xi_zero;
  /* Fourier vector on device */
  float *dev_xi_zero;
  /* right hand side tau on host */
  float *tau;
  /* right hand side tau on host */
  cufftComplex *dev_tau;
  /* right hand side tau on device */
  cufftComplex *dev_tau_hat;
  /* Fourier-transform of tau */
  cufftComplex *tau_hat;
  /* Fourier transform of strain tensor on device */
  cufftComplex *dev_epsilon_hat;
  /* train tensor on device */
  cufftComplex *dev_epsilon;
  /* Fourier transform of strain tensor on host */
  cufftComplex *epsilon_hat;
  /* Strain tensor on host */
  float *epsilon;
  /* Stress tensor on host */
  float *sigma;
  /* Divergence of stress tensor on host */
  float *div_sigma;
  /* Random number generator */
  std::default_random_engine rng;
  /* normal distribution */
  std::normal_distribution<float> distribution;
  /* cuFFT plan */
  cufftHandle plan;
};

/** @brief Check whether xi-zero is constructed consistently on device and host
 */
TEST_F(FourierTest, TestXiZero)
{
  float tolerance = 1.E-6;
  // halo size
  size_t nvoxels = grid_spec.number_of_voxels();
  // allocate host memory
  float *xi_zero_ref = nullptr;
  CUDA_CHECK(cudaMallocHost(&xi_zero_ref, 3 * nvoxels * sizeof(float)));
  initialize_xizero_host(xi_zero_ref, grid_spec);

  CUDA_CHECK(cudaMemcpy(xi_zero, dev_xi_zero, 3 * nvoxels * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float rel_diff = relative_difference(xi_zero, xi_zero_ref, 3 * nvoxels);

  // Free memory
  CUDA_CHECK(cudaFreeHost(xi_zero_ref));

  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/* Check whether divergence computation is consistent in Fourier- and real space
 */
TEST_F(FourierTest, TestFourierDivergence)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  float *dev_xi = nullptr;
  float *div_epsilon = nullptr;
  cufftComplex *dev_div_epsilon_hat = nullptr;
  cufftComplex *div_epsilon_hat = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_xi, 3 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_div_epsilon_hat, 3 * nvoxels * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&div_epsilon, 3 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&div_epsilon_hat, 3 * nvoxels * sizeof(cufftComplex)));

  // Initialize epsilon with random numbers
  std::generate(epsilon, epsilon + 6 * nvoxels, [&]()
                { return distribution(rng); });

  // compute divergence of epsilon in real space
  backward_divergence_host(epsilon, div_epsilon, grid_spec);

  CUDA_CHECK(cudaMemset(dev_epsilon, 0, 6 * nvoxels * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMemcpy2D(dev_epsilon, 2 * sizeof(float), epsilon, sizeof(float), sizeof(float), 6 * nvoxels, cudaMemcpyHostToDevice));

  // Fourier transform epsilon
  CUFFT_CHECK(cufftExecC2C(plan, dev_epsilon, dev_epsilon_hat, CUFFT_FORWARD));
  CUDA_CHECK(cudaDeviceSynchronize());

  // compute divergence of hat(epsilon) in Fourier space on device
  initialize_xi_device(dev_xi, grid_spec);
  divergence_fourier(dev_epsilon_hat, dev_div_epsilon_hat, dev_xi, grid_spec);
  cudaDeviceSynchronize();
  // Copy back to host
  CUDA_CHECK(cudaMemcpy(div_epsilon_hat, dev_div_epsilon_hat, 3 * nvoxels * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  // Compute norms ||D(epsilon)|| and ||xi.hat(epsilon)||
  float norm_real = vector_norm(div_epsilon, nvoxels);
  float norm_fourier = vector_norm(div_epsilon_hat, nvoxels) / sqrt(nvoxels);
  float rel_diff = (norm_fourier - norm_real) / norm_real;
  // Free memory
  CUDA_CHECK(cudaFreeHost(div_epsilon));
  CUDA_CHECK(cudaFree(dev_xi));
  CUDA_CHECK(cudaFree(dev_div_epsilon_hat));
  CUDA_CHECK(cudaFreeHost(div_epsilon_hat));
  float tolerance = 1.E-4;
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
 */
TEST_F(FourierTest, TestDivSigma)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  // Initialize tau with random numbers
  std::generate(tau, tau + 6 * nvoxels, [&]()
                { return distribution(rng); });
  CUDA_CHECK(cudaMemset(dev_tau, 0, 6 * nvoxels * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMemcpy2D(dev_tau, 2 * sizeof(float), tau, sizeof(float), sizeof(float), 6 * nvoxels, cudaMemcpyHostToDevice));

  // Fourier transform the real-valued tau
  CUFFT_CHECK(cufftExecC2C(plan, dev_tau, dev_tau_hat, CUFFT_FORWARD));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Solve on device and Fourier transform back
  fourier_solve_device(dev_tau_hat, dev_epsilon_hat, dev_xi_zero,
                       lambda_0, mu_0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUFFT_CHECK(cufftExecC2C(plan, dev_epsilon_hat, dev_epsilon, CUFFT_INVERSE));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back to host
  CUDA_CHECK(cudaMemcpy2D(epsilon, sizeof(float), dev_epsilon, 2 * sizeof(float), sizeof(float), 6 * nvoxels, cudaMemcpyDeviceToHost));

  // Scale by 1/nvoxels to account for the fact that the inverse Fourier transform is not
  // normalised in cuFFT
  for (int ell = 0; ell < 6 * nvoxels; ++ell)
    epsilon[ell] *= 1.0 / nvoxels;
  // Compute stress sigma_{ij} = C^0_{ijkl} epsilon_{kl} + tau_{ij}
  for (int ell = 0; ell < nvoxels; ++ell)
  {
    float tr_epsilon = epsilon[0 * nvoxels + ell] + epsilon[1 * nvoxels + ell] + epsilon[2 * nvoxels + ell];
    for (int alpha = 0; alpha < 6; ++alpha)
    {
      int idx = alpha * nvoxels + ell;
      sigma[idx] = tau[idx] + 2 * mu_0 * epsilon[idx];
      if (alpha < 3)
      {
        sigma[idx] += lambda_0 * tr_epsilon;
      }
    }
  }
  // Compute divergence
  backward_divergence_host(sigma, div_sigma, grid_spec);

  float sigma_nrm = tensor_norm(sigma, nvoxels);
  float div_nrm = vector_norm(div_sigma, nvoxels);
  float rel_diff = div_nrm / sigma_nrm;
  float tolerance = 5.E-5;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

#endif // TEST_FOURIER_HH