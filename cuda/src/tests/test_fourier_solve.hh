#ifndef TEST_FOURIER_SOLVE_HH
#define TEST_FOURIER_SOLVE_HH TEST_FOURIER_SOLVE_HH
#include "fourier_solve.hh"
#include <gtest/gtest.h>

class FourierSolveTest : public ::testing::Test {
public:
  /** @Create a new instance */
  FourierSolveTest() {}

protected:
  /** @brief initialise tests */
  void SetUp() override {}
};

/** @brief Check whether xi-zero is constructed consistently on device and host
 */
TEST_F(FourierSolveTest, TestXiZero) {
  float tolerance = 1.E-6;
  // halo size
  GridSpec grid_spec;
  grid_spec.nx = 48;
  grid_spec.ny = 64;
  grid_spec.nz = 32;
  grid_spec.Lx = 1.1;
  grid_spec.Ly = 0.9;
  grid_spec.Lz = 0.7;
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
                        cudaMemcpyDefault));

  float rel_diff = relative_difference(xi_zero, xi_zero_ref, 3 * ncells);

  // Free memory
  CUDA_CHECK(cudaFreeHost(xi_zero));
  CUDA_CHECK(cudaFreeHost(xi_zero_ref));
  CUDA_CHECK(cudaFree(dev_xi_zero));

  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}
#endif // TEST_FOURIER_SOLVE_HH