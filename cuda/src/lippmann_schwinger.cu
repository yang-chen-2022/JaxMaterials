#include "lippmann_schwinger.hh"

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