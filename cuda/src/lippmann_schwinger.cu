#include "lippmann_schwinger.hh"

/* Kernel for computing stress sigma_{ij} = C_{ijkl} epsilon_{kl} with
 *
 *     C_{ijkl} = lambda*delta_{ij}delta_{kl} + mu*(delta_{ik}delta_{jl}+delta_{il}delta_{jk})
 */
__global__ void compute_stress_kernel(float *dev_epsilon, float *dev_sigma,
                                      float *dev_lambda, float *dev_mu,
                                      const int ncells)
{
  int ell = blockDim.x * blockIdx.x + threadIdx.x;
  if (ell < ncells)
  {
    float lambda = dev_lambda[ell];
    float mu = dev_mu[ell];
    float tr_epsilon = dev_epsilon[ell] + dev_epsilon[ncells + ell] + dev_epsilon[2 * ncells + ell];
    for (int alpha = 0; alpha < 3; ++alpha)
    {
      int idx = alpha * ncells + ell;
      dev_sigma[idx] = 2 * mu * dev_epsilon[idx] + lambda * tr_epsilon;
    }
    for (int alpha = 3; alpha < 6; ++alpha)
    {
      int idx = alpha * ncells + ell;
      dev_sigma[idx] = 2 * mu * dev_epsilon[idx];
    }
  }
}

/* Compute stress sigma_{ij} = C_{ijkl} epsilon_{kl} on device */
void compute_stress(float *dev_epsilon, float *dev_sigma,
                    float *dev_lambda, float *dev_mu,
                    const GridSpec grid_spec)
{
  int ncells = grid_spec.number_of_cells();
  const int nblocks = (ncells + BLOCKSIZE - 1) / BLOCKSIZE;
  compute_stress_kernel<<<nblocks, BLOCKSIZE>>>(dev_epsilon, dev_sigma,
                                                dev_lambda, dev_mu, ncells);
}

/* Kernel for incrementing solution epsilon -> epsilon + alpha*r */
__global__ void increment_solution_kernel(float *dev_epsilon, cufftComplex *dev_r,
                                          const float alpha,
                                          const int ndof)
{
  int ell = blockDim.x * blockIdx.x + threadIdx.x;
  if (ell < ndof)
  {
    dev_epsilon[ell] += alpha * dev_r[ell].x;
  }
}

/* Increment solution epsilon -> epsilon + 1/ncells * r */
void increment_solution(float *dev_epsilon, cufftComplex *dev_r,
                        const GridSpec grid_spec)
{
  int ncells = grid_spec.number_of_cells();
  int ndof = 6 * ncells;
  const int nblocks = (ndof + BLOCKSIZE - 1) / BLOCKSIZE;
  float alpha = 1 / ncells;
  increment_solution_kernel<<<nblocks, BLOCKSIZE>>>(dev_epsilon, dev_r, alpha, ndof);
}

/* Lippmann Schwinger iteration */
extern "C"
{
  void lippmann_schwinger_solve(float *lambda, float *mu, float *epsilon_bar,
                                float *epsilon, float *sigma,
                                int *cells,
                                float *extents)
  {
    GridSpec grid_spec;
    grid_spec.nx = cells[0];
    grid_spec.ny = cells[1];
    grid_spec.nz = cells[2];
    grid_spec.Lx = extents[0];
    grid_spec.Ly = extents[1];
    grid_spec.Lz = extents[2];

    int ncells = grid_spec.number_of_cells();
    // Set up cuFFT plan
    cufftHandle plan;
    cufftComplex *data;
    CUDA_CHECK(cudaMalloc(&data, 6 * ncells * sizeof(cufftComplex)));
    int n[3] = {grid_spec.nz, grid_spec.ny, grid_spec.nx};
    CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, ncells, n, 1, ncells, CUFFT_C2C, 6));
    // Average values of lambda and mu
    float lambda_0 = 0.5 * (*std::max_element(lambda, lambda + ncells) + *std::min_element(lambda, lambda + ncells));
    float mu_0 = 0.5 * (*std::min_element(lambda, lambda + ncells) + *std::min_element(mu, mu + ncells));
    // allocate device memory
    float *dev_lambda = nullptr;           // Lame parameter lamba on device
    float *dev_mu = nullptr;               // Lame parameter mu on device
    float *dev_epsilon = nullptr;          // real-valued strain epsilon on device
    float *dev_sigma = nullptr;            // real-valued stress sigma on device
    cufftComplex *dev_sigma_hat = nullptr; // complex-valued Fourier-stress on device
    cufftComplex *dev_residual = nullptr;  // complex-valued residual on device
    float *dev_xi_zero = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * ncells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_lambda, 6 * ncells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_mu, 6 * ncells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_epsilon, 6 * ncells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_sigma, 6 * ncells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_sigma_hat, 6 * ncells * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&dev_residual, 6 * ncells * sizeof(cufftComplex)));
    // copy Lame parameters to device
    CUDA_CHECK(cudaMemcpy(dev_lambda, lambda, ncells * sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(dev_mu, mu, ncells * sizeof(float), cudaMemcpyDefault));
    // set average value of epsilon
    for (int alpha = 0; alpha < 6; ++alpha)
      CUDA_CHECK(cudaMemset(dev_epsilon + alpha * ncells, epsilon_bar[alpha], ncells * sizeof(float)));
    // initialize Fourier vectors
    initialize_xizero(dev_xi_zero, grid_spec);
    CUDA_CHECK(cudaDeviceSynchronize());

    // main Lippmann-Schwinger loop
    const int maxiter = 10;
    for (int iter = 0; iter < maxiter; ++iter)
    {
      printf("iteration %4d\n", iter);
      /* ==== STEP 1 ==== Compute stress: sigma_{ij} = C_{ijkl} epsilon_{kl} */
      compute_stress(dev_epsilon, dev_sigma, dev_lambda, dev_mu, grid_spec);
      /* ==== STEP 2 ==== Check convergence */
      // TODO
      if (false)
      {
        break;
      }
      /* ==== STEP 3 ==== Fourier transform:  hat(sigma) = FFT(sigma)*/
      CUDA_CHECK(cudaMemset(dev_sigma_hat, 0, 6 * ncells * sizeof(cufftComplex)));
      CUDA_CHECK(cudaMemcpy2D(dev_sigma_hat, 2 * sizeof(float), dev_sigma, sizeof(float), sizeof(float), 6 * ncells, cudaMemcpyDefault));
      CUFFT_CHECK(cufftExecC2C(plan, dev_sigma_hat, dev_sigma_hat, CUFFT_FORWARD));
      CUDA_CHECK(cudaDeviceSynchronize());
      /* ==== STEP 4 ==== Solve in Fourier space: hat(r)_{kl} = -Gamma^{0}_{klij} hat(sigma)_{ij} */
      fourier_solve_device(dev_sigma_hat, dev_residual, dev_xi_zero, lambda_0, mu_0, grid_spec);
      CUDA_CHECK(cudaDeviceSynchronize());
      /* ==== STEP 5 ==== Inverse Fourier transform: r = FFT^{-1}(hat(r)) */
      CUFFT_CHECK(cufftExecC2C(plan, dev_residual, dev_residual, CUFFT_INVERSE));
      CUDA_CHECK(cudaDeviceSynchronize());
      /* ==== STEP 6 ==== Update solution: epsilon -> epsilon + r*/
      increment_solution(dev_epsilon, dev_residual, grid_spec);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    // Copy solution back to host
    CUDA_CHECK(cudaMemcpy(epsilon, dev_epsilon, 6 * ncells * sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(sigma, dev_sigma, 6 * ncells * sizeof(float), cudaMemcpyDefault));

    // free memory
    CUDA_CHECK(cudaFree(dev_xi_zero));
    CUDA_CHECK(cudaFree(dev_lambda));
    CUDA_CHECK(cudaFree(dev_mu));
    CUDA_CHECK(cudaFree(dev_epsilon));
    CUDA_CHECK(cudaFree(dev_sigma));
    CUDA_CHECK(cudaFree(dev_residual));
    CUDA_CHECK(cudaFree(dev_sigma_hat));
  }
}