#include "common.hh"
#include "lippmann_schwinger.hh"
#include "profile.hh"

int main()
{
  // domain size
  profile_derivatives();
  int cells[3] = {64, 64, 64};
  float extents[3] = {1.0, 1.0, 1.0};
  int ncells = cells[0] * cells[1] * cells[2];
  float *lambda;
  float *mu;
  float epsilon_bar[6] = {1.0, 0.4, 0.2, 1.5, 0.8, 0.7};
  float *epsilon;
  float *sigma;
  CUDA_CHECK(cudaMallocHost(&lambda, ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&mu, ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon, 6 * ncells * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * ncells * sizeof(float)));

  lippmann_schwinger_solve(lambda, mu, epsilon_bar, epsilon, sigma, cells, extents);

  CUDA_CHECK(cudaFreeHost(lambda));
  CUDA_CHECK(cudaFreeHost(mu));
  CUDA_CHECK(cudaFreeHost(epsilon));
  CUDA_CHECK(cudaFreeHost(sigma));
}