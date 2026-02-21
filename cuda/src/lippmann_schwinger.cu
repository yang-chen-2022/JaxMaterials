#include "lippmann_schwinger.hh"

void lippmann_schwinger_solve(const GridSpec grid_spec)
{
  // halo size

  int domain_volume = grid_spec.number_of_cells();
  // allocate host memory
  float *u = nullptr;
  float *du_dx = nullptr;
  float *du_dx_ref = nullptr;
  cudaMallocHost(&u, domain_volume * sizeof(float));
  cudaMallocHost(&du_dx, domain_volume * sizeof(float));
  cudaMallocHost(&du_dx_ref, domain_volume * sizeof(float));

  // initialise data
  init_field(u, grid_spec);

  // allocate device memory
  float *dev_u = nullptr;
  float *dev_du_dx = nullptr;
  cudaMalloc(&dev_u, domain_volume * sizeof(float));
  cudaMalloc(&dev_du_dx, domain_volume * sizeof(float));

  cudaMemcpy(dev_u, u, domain_volume * sizeof(float), cudaMemcpyDefault);
  backward_derivative_device(dev_u, dev_du_dx, 0, grid_spec);
  cudaDeviceSynchronize();
  cudaMemcpy(du_dx, dev_du_dx, domain_volume * sizeof(float), cudaMemcpyDefault);
  backward_derivative_host(u, du_dx_ref, 0, grid_spec);
  float rel_diff = relative_difference(du_dx, du_dx_ref, grid_spec);
  printf("Relative difference = %8.4e\n", rel_diff);
  cudaFree(dev_u);
  cudaFree(dev_du_dx);
  cudaFreeHost(u);
  cudaFreeHost(du_dx);
  cudaFreeHost(du_dx_ref);
}