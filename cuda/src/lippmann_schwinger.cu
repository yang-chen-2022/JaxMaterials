#include "lippmann_schwinger.hh"

/* **** CUDA kernels **** */

/* Kernel for setting the values of epsilon to the constant bar(epsilon) */
__global__ void set_epsilon_bar_kernel(float *dev_epsilon, float *epsilon_bar, const int ncells)
{
  int ell = blockDim.x * blockIdx.x + threadIdx.x;
  if (ell < ncells)
    for (int alpha = 0; alpha < 6; ++alpha)
      dev_epsilon[alpha * ncells + ell] = alpha;
}

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

/* **** class methods **** */

/* Set the values of epsilon to bar(epsilon) on the device */
void LippmannSchwingerSolver::set_epsilon_bar(float *dev_epsilon, float *epsilon_bar)
{
  int ncells = grid_spec.number_of_cells();
  const int nblocks = (ncells + BLOCKSIZE - 1) / BLOCKSIZE;
  set_epsilon_bar_kernel<<<nblocks, BLOCKSIZE>>>(dev_epsilon, epsilon_bar, ncells);
}

/* Compute stress sigma_{ij} = C_{ijkl} epsilon_{kl} on device */
void LippmannSchwingerSolver::compute_stress(float *dev_epsilon, float *dev_sigma,
                                             float *dev_lambda, float *dev_mu)
{
  int ncells = grid_spec.number_of_cells();
  const int nblocks = (ncells + BLOCKSIZE - 1) / BLOCKSIZE;
  compute_stress_kernel<<<nblocks, BLOCKSIZE>>>(dev_epsilon, dev_sigma,
                                                dev_lambda, dev_mu, ncells);
}

/* Increment solution epsilon -> epsilon + 1/ncells * r */
void LippmannSchwingerSolver::increment_solution(float *dev_epsilon, cufftComplex *dev_r)
{
  int ncells = grid_spec.number_of_cells();
  int ndof = 6 * ncells;
  const int nblocks = (ndof + BLOCKSIZE - 1) / BLOCKSIZE;
  float alpha = 1.0 / float(ncells);
  increment_solution_kernel<<<nblocks, BLOCKSIZE>>>(dev_epsilon, dev_r, alpha, ndof);
}

/* Compute normalised divergence for stopping criterion in Fourier space */
float LippmannSchwingerSolver::relative_divergence_norm(cufftComplex *dev_sigma_hat)
{
  // Compute divergence in Fourier space
  divergence_fourier(dev_sigma_hat, dev_div_sigma_hat, dev_xi, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  int ncells = grid_spec.number_of_cells();
  // STEP 1: Compute nrm_div_sigma =  <||div(sigma)||^2>
  float nrm_div_sigma = 0;
  CUBLAS_CHECK(cublasScnrm2(handle, 3 * ncells, dev_div_sigma_hat, 1, &nrm_div_sigma));
  // Scale by number of cells
  nrm_div_sigma /= sqrt(ncells);
  // STEP 2: compute ||<sigma>||^2
  // Extract zero mode, which is identical to the sum of sigma over the domain, i.e. ncells * <sigma>
  CUDA_CHECK(cudaMemcpy2D(sigma_0, sizeof(cufftComplex), dev_sigma_hat, ncells * sizeof(cufftComplex), sizeof(cufftComplex), 6, cudaMemcpyDeviceToHost));
  // Normalise by number of cells
  for (int alpha = 0; alpha < 6; ++alpha)
  {
    sigma_0[alpha].x /= ncells;
    sigma_0[alpha].y /= ncells;
  }
  // Compute norm of zero mode

  float nrm_sigma = tensor_norm(sigma_0, 1);
  return nrm_div_sigma / nrm_sigma;
}

/* Constructor */
LippmannSchwingerSolver::LippmannSchwingerSolver(const GridSpec grid_spec, const int verbose) : grid_spec(grid_spec), verbose(verbose)
{
  int ncells = grid_spec.number_of_cells();
  // Initialise cuBLAS
  CUBLAS_CHECK(cublasCreate(&handle));
  // Set up cuFFT plan
  int n[3] = {grid_spec.nz, grid_spec.ny, grid_spec.nx};
  CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, ncells, n, 1, ncells, CUFFT_C2C, 6));
  CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_xi, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_lambda, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_mu, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_epsilon, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_sigma, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_div_sigma, 3 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_div_sigma_hat, 3 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_sigma_hat, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_residual, 6 * ncells * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&sigma_0, 6 * sizeof(cufftComplex)));
  // initialize Fourier vectors
  initialize_xi(dev_xi, grid_spec);
  initialize_xizero(dev_xi_zero, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
}

/* Destructor */
LippmannSchwingerSolver::~LippmannSchwingerSolver()
{
  // free memory
  CUDA_CHECK(cudaFree(dev_xi));
  CUDA_CHECK(cudaFree(dev_xi_zero));
  CUDA_CHECK(cudaFree(dev_lambda));
  CUDA_CHECK(cudaFree(dev_mu));
  CUDA_CHECK(cudaFree(dev_epsilon));
  CUDA_CHECK(cudaFree(dev_sigma));
  CUDA_CHECK(cudaFree(dev_div_sigma));
  CUDA_CHECK(cudaFree(dev_div_sigma_hat));
  CUDA_CHECK(cudaFree(dev_residual));
  CUDA_CHECK(cudaFree(dev_sigma_hat));
  CUDA_CHECK(cudaFreeHost(sigma_0));
  CUBLAS_CHECK(cublasDestroy(handle));
}

/* apply solver */
int LippmannSchwingerSolver::apply(float *lambda, float *mu, float *epsilon_bar,
                                   float *epsilon, float *sigma,
                                   float tolerance, int maxiter)
{
  int ncells = grid_spec.number_of_cells();
  // Average values of lambda and mu
  float lambda_0 = 0.5 * (*std::max_element(lambda, lambda + ncells) + *std::min_element(lambda, lambda + ncells));
  float mu_0 = 0.5 * (*std::min_element(lambda, lambda + ncells) + *std::min_element(mu, mu + ncells));

  // copy Lame parameters to device
  CUDA_CHECK(cudaMemcpy(dev_lambda, lambda, ncells * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_mu, mu, ncells * sizeof(float), cudaMemcpyHostToDevice));

  // set average value of epsilon
  set_epsilon_bar(dev_epsilon, epsilon_bar);
  CUDA_CHECK(cudaDeviceSynchronize());
  // main Lippmann-Schwinger loop
  float rel_div_norm = 0;
  int iter;
  if (verbose == 2)
  {
    printf("==== Lippmann Schwinger solver ====\n");
    printf("  iteration    rel. residual\n");
  }
  for (iter = 0; iter < maxiter; ++iter)
  {
    /* ==== STEP 1 ==== Compute stress: sigma_{ij} = C_{ijkl} epsilon_{kl} */
    compute_stress(dev_epsilon, dev_sigma, dev_lambda, dev_mu);
    /* ==== STEP 2 ==== Fourier transform:  hat(sigma) = FFT(sigma)*/
    CUDA_CHECK(cudaMemset(dev_sigma_hat, 0, 6 * ncells * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy2D(dev_sigma_hat, 2 * sizeof(float), dev_sigma, sizeof(float), sizeof(float), 6 * ncells, cudaMemcpyDeviceToDevice));
    CUFFT_CHECK(cufftExecC2C(plan, dev_sigma_hat, dev_sigma_hat, CUFFT_FORWARD));
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 3 ==== Check convergence */
    rel_div_norm = relative_divergence_norm(dev_sigma_hat);
    if (verbose > 1)
      printf("     %4d          %8.4e\n", iter, rel_div_norm);
    if (rel_div_norm < tolerance)
      break;
    /* ==== STEP 4 ==== Solve in Fourier space: hat(r)_{kl} = -Gamma^{0}_{klij} hat(sigma)_{ij} */
    fourier_solve_device(dev_sigma_hat, dev_residual, dev_xi_zero, lambda_0, mu_0, grid_spec);
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 5 ==== Inverse Fourier transform: r = FFT^{-1}(hat(r)) */
    CUFFT_CHECK(cufftExecC2C(plan, dev_residual, dev_residual, CUFFT_INVERSE));
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 6 ==== Update solution: epsilon -> epsilon + r*/
    increment_solution(dev_epsilon, dev_residual);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  if (verbose > 0)
  {
    if (verbose == 1)
      printf("  Lippmann Schwinger solver ");
    else
      printf("  ");
    if (iter < maxiter)
      printf("converged");
    else
      printf("failed to converge");
    printf(" after %4d its, rel. res. = %8.4e\n", iter, rel_div_norm);
  }
  // Copy solution back to host
  CUDA_CHECK(cudaMemcpy(epsilon, dev_epsilon, 6 * ncells * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sigma, dev_sigma, 6 * ncells * sizeof(float), cudaMemcpyDeviceToHost));

  return iter;
}
/* Lippmann Schwinger iteration */
extern "C"
{
  int lippmann_schwinger_solve(float *lambda, float *mu, float *epsilon_bar,
                               float *epsilon, float *sigma,
                               int *cells,
                               float *extents,
                               float tolerance, int maxiter)
  {
    GridSpec grid_spec;
    grid_spec.nx = cells[0];
    grid_spec.ny = cells[1];
    grid_spec.nz = cells[2];
    grid_spec.Lx = extents[0];
    grid_spec.Ly = extents[1];
    grid_spec.Lz = extents[2];
    LippmannSchwingerSolver solver(grid_spec);
    int iter = solver.apply(lambda, mu, epsilon_bar,
                            epsilon, sigma,
                            tolerance, maxiter);
    return iter;
  }
}