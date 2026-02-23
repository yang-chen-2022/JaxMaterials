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
  float tr_epsilon;
  if (ell < ncells)
  {
    tr_epsilon = dev_epsilon[ell] + dev_epsilon[ncells + ell] + dev_epsilon[2 * ncells + ell];
    for (int mu = 0; mu < 3; ++mu)
    {
      int idx = mu * ncells + ell;
      dev_sigma[idx] = 2 * dev_mu[idx] * dev_epsilon[idx] + dev_lambda[idx] * tr_epsilon;
    }
    for (int mu = 3; mu < 6; ++mu)
    {
      int idx = mu * ncells + ell;
      dev_sigma[idx] = 2 * dev_mu[idx] * dev_epsilon[idx];
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

void lippmann_schwinger_solve(const GridSpec grid_spec)
{
  // halo size

  int ncells = grid_spec.number_of_cells();

  // allocate device memory
  float *dev_u = nullptr;
  float *dev_v = nullptr;
  float *dev_xi_zero = nullptr;
  cudaMalloc(&dev_xi_zero, 3 * ncells * sizeof(float));
  initialize_xizero(dev_xi_zero, grid_spec);
  cudaDeviceSynchronize();
  cudaFree(dev_xi_zero);
}