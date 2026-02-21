#include "lippmann_schwinger.hh"

void lippmann_schwinger_solve(const GridSpec grid_spec)
{
  // halo size

  int domain_volume = grid_spec.number_of_cells();

  // allocate host memory
  float *u = nullptr;
  float *v = nullptr;
  cudaMallocHost(&u, domain_volume * sizeof(float));
  cudaMallocHost(&v, domain_volume * sizeof(float));

  // allocate device memory
  float *dev_u = nullptr;
  float *dev_v = nullptr;
  float *dev_xi_zero_0 = nullptr;
  float *dev_xi_zero_1 = nullptr;
  float *dev_xi_zero_2 = nullptr;
  cudaMalloc(&dev_u, domain_volume * sizeof(float));
  cudaMalloc(&dev_v, domain_volume * sizeof(float));
  cudaMalloc(&dev_xi_zero_0, domain_volume * sizeof(float));
  cudaMalloc(&dev_xi_zero_1, domain_volume * sizeof(float));
  cudaMalloc(&dev_xi_zero_2, domain_volume * sizeof(float));
  initialize_xizero(dev_xi_zero_0, dev_xi_zero_1, dev_xi_zero_2, grid_spec);
  cudaDeviceSynchronize();
  fourier_solve(dev_u, dev_v, grid_spec);
  cudaDeviceSynchronize();
  cudaMemcpy(u, dev_u, domain_volume * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(v, dev_v, domain_volume * sizeof(float), cudaMemcpyDefault);
  for (int i = 0; i < grid_spec.nx; ++i)
  {
    printf("%8.4f ", u[IDX(grid_spec.nx, grid_spec.ny, grid_spec.nz, i, 0, 0)]);
  }
  printf("\n");
  cudaFree(dev_u);
  cudaFree(dev_v);
  cudaFree(dev_xi_zero_0);
  cudaFree(dev_xi_zero_1);
  cudaFree(dev_xi_zero_2);
  cudaFreeHost(u);
  cudaFreeHost(v);
}