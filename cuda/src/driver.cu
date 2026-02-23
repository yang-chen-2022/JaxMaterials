#include "common.hh"
#include "lippmann_schwinger.hh"
#include "profile.hh"

int main()
{
  // domain size
  profile_derivatives();
  GridSpec grid_spec;
  grid_spec.nx = 64;
  grid_spec.ny = 64;
  grid_spec.nz = 64;
  grid_spec.Lx = 1.0;
  grid_spec.Ly = 1.0;
  grid_spec.Lz = 1.0;
  int ncells = grid_spec.number_of_cells();

  float *lambda;
  float *mu;
  float epsilon_bar[6] = {1.0, 0.4, 0.2, 1.5, 0.8, 0.7};
  float *epsilon;
  float *sigma;
  CUDA_CHECK(cudaMallocHost(&lambda, ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&mu, ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * ncells * sizeof(float)));

  lippmann_schwinger_solve(lambda, mu, epsilon_bar, epsilon, sigma, grid_spec);

  CUDA_CHECK(cudaFreeHost(lambda));
  CUDA_CHECK(cudaFreeHost(mu));
  CUDA_CHECK(cudaFreeHost(epsilon));
  CUDA_CHECK(cudaFreeHost(sigma));
}