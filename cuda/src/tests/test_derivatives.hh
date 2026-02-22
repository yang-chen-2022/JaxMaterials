#ifndef TEST_DERIVATIVES_HH
#define TEST_DERIVATIVES_HH TEST_DERIVATIVES_HH
#include "derivatives.hh"
#include <gtest/gtest.h>

/** @brief test derivatives
 */
/** @brief test derivatives
 */
class DerivativeTest : public ::testing::Test
{
public:
    /** @Create a new instance */
    DerivativeTest() {}

protected:
    /** @brief initialise tests */
    void SetUp() override {}
    /** Test */
    void test_derivative(const int direction, const bool use_shared_memory)
    {
        float tolerance = 1.E-6;
        // halo size
        GridSpec grid_spec;
        grid_spec.nx = 48;
        grid_spec.ny = 64;
        grid_spec.nz = 32;
        grid_spec.Lx = 1.1;
        grid_spec.Ly = 0.9;
        grid_spec.Lz = 0.7;
        int domain_volume = grid_spec.number_of_cells();
        // allocate host memory
        float *u = nullptr;
        float *du_dx = nullptr;
        float *du_dx_ref = nullptr;
        CUDA_CHECK(cudaMallocHost(&u, domain_volume * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&du_dx, domain_volume * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&du_dx_ref, domain_volume * sizeof(float)));

        // initialise data
        init_field(u, grid_spec);

        // allocate device memory
        float *dev_u = nullptr;
        float *dev_du_dx = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_u, domain_volume * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_du_dx, domain_volume * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dev_u, u, domain_volume * sizeof(float), cudaMemcpyDefault));

        backward_derivative_device(dev_u, dev_du_dx, direction, grid_spec,
                                   use_shared_memory);
        CUDA_CHECK(cudaDeviceSynchronize());
        backward_derivative_host(u, du_dx_ref, direction, grid_spec);

        CUDA_CHECK(cudaMemcpy(du_dx, dev_du_dx, domain_volume * sizeof(float),
                              cudaMemcpyDefault));
        int ncells = grid_spec.number_of_cells();
        float rel_diff = relative_difference(du_dx, du_dx_ref, ncells);
        CUDA_CHECK(cudaFree(dev_u));
        CUDA_CHECK(cudaFree(dev_du_dx));
        CUDA_CHECK(cudaFreeHost(u));
        CUDA_CHECK(cudaFreeHost(du_dx));
        CUDA_CHECK(cudaFreeHost(du_dx_ref));
        EXPECT_NEAR(rel_diff, 0.0, tolerance);
    }
};

/** @brief Check whether derivative in x-direction is correct
 */
TEST_F(DerivativeTest, TestXDerivative) { test_derivative(0, false); }

/** @brief Check whether derivative in y-direction is correct
 */
TEST_F(DerivativeTest, TestYDerivative) { test_derivative(1, false); }

/** @brief Check whether derivative in z-direction is correct
 */
TEST_F(DerivativeTest, TestZDerivative) { test_derivative(2, false); }

/** @brief Check whether derivative in x-direction is correct
 */
TEST_F(DerivativeTest, TestXDerivativeShared) { test_derivative(0, true); }

/** @brief Check whether derivative in y-direction is correct
 */
TEST_F(DerivativeTest, TestYDerivativeShared) { test_derivative(1, true); }

/** @brief Check whether derivative in z-direction is correct
 */
TEST_F(DerivativeTest, TestZDerivativeShared) { test_derivative(2, true); }

#endif // TEST_DERIVATIVES_HH