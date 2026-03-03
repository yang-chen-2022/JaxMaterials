#ifndef TEST_DERIVATIVES_HH
#define TEST_DERIVATIVES_HH TEST_DERIVATIVES_HH
#include <random>
#include <algorithm>
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
    void SetUp() override
    {
        grid_spec.nx = 48;
        grid_spec.ny = 64;
        grid_spec.nz = 32;
        grid_spec.Lx = 1.1;
        grid_spec.Ly = 0.9;
        grid_spec.Lz = 0.7;
        tolerance = 1.E-6;
        rng.seed(7812481);
    }
    /* test backward derivative in a particular direction */
    void test_derivative(const int direction)
    {
        size_t nvoxels = grid_spec.number_of_voxels();
        // allocate host memory
        float *u = nullptr;
        float *du_dx = nullptr;
        float *du_dx_ref = nullptr;
        CUDA_CHECK(cudaMallocHost(&u, nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&du_dx, nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&du_dx_ref, nvoxels * sizeof(float)));

        // initialise data
        std::generate(u, u + nvoxels, [&]()
                      { return distribution(rng); });
        // allocate device memory
        float *dev_u = nullptr;
        float *dev_du_dx = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_u, nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_du_dx, nvoxels * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dev_u, u, nvoxels * sizeof(float), cudaMemcpyDefault));

        backward_derivative_device(dev_u, dev_du_dx, direction, grid_spec);
        CUDA_CHECK(cudaDeviceSynchronize());
        backward_derivative_host(u, du_dx_ref, direction, grid_spec);

        CUDA_CHECK(cudaMemcpy(du_dx, dev_du_dx, nvoxels * sizeof(float),
                              cudaMemcpyDefault));
        float rel_diff = relative_difference(du_dx, du_dx_ref, nvoxels);
        CUDA_CHECK(cudaFree(dev_u));
        CUDA_CHECK(cudaFree(dev_du_dx));
        CUDA_CHECK(cudaFreeHost(u));
        CUDA_CHECK(cudaFreeHost(du_dx));
        CUDA_CHECK(cudaFreeHost(du_dx_ref));
        EXPECT_NEAR(rel_diff, 0.0, tolerance);
    }

    /* test backward divergence*/
    void test_divergence()
    {
        size_t nvoxels = grid_spec.number_of_voxels();
        // allocate host memory
        float *sigma = nullptr;
        float *div_sigma = nullptr;
        float *div_sigma_ref = nullptr;
        CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&div_sigma, 3 * nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&div_sigma_ref, 3 * nvoxels * sizeof(float)));

        // initialise data
        std::generate(sigma, sigma + 6 * nvoxels, [&]()
                      { return distribution(rng); });
        // allocate device memory
        float *dev_sigma = nullptr;
        float *dev_div_sigma = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_sigma, 6 * nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_div_sigma, 3 * nvoxels * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dev_sigma, sigma, 6 * nvoxels * sizeof(float), cudaMemcpyDefault));

        backward_divergence_device(dev_sigma, dev_div_sigma, grid_spec);
        CUDA_CHECK(cudaDeviceSynchronize());
        backward_divergence_host(sigma, div_sigma_ref, grid_spec);

        CUDA_CHECK(cudaMemcpy(div_sigma, dev_div_sigma, 3 * nvoxels * sizeof(float),
                              cudaMemcpyDefault));
        float rel_diff = relative_difference(div_sigma, div_sigma_ref, 3 * nvoxels);
        CUDA_CHECK(cudaFree(dev_sigma));
        CUDA_CHECK(cudaFree(dev_div_sigma));
        CUDA_CHECK(cudaFreeHost(sigma));
        CUDA_CHECK(cudaFreeHost(div_sigma));
        CUDA_CHECK(cudaFreeHost(div_sigma_ref));
        EXPECT_NEAR(rel_diff, 0.0, tolerance);
    }

    /* Class variables */
    GridSpec grid_spec;                           // grid specification
    float tolerance;                              // tolerance
    std::default_random_engine rng;               // random number generator
    std::normal_distribution<float> distribution; // random number distribution used for initialisation
};

/** @brief Check whether derivative in x-direction agrees between device and host
 */
TEST_F(DerivativeTest, TestXDerivative) { test_derivative(0); }

/** @brief Check whether derivative in y-direction agrees between device and host
 */
TEST_F(DerivativeTest, TestYDerivative) { test_derivative(1); }

/** @brief Check whether derivative in z-direction agrees between device and host
 */
TEST_F(DerivativeTest, TestZDerivative) { test_derivative(2); }

/** @brief Check whether divergence agrees between device and host
 */
TEST_F(DerivativeTest, TestDivergence) { test_divergence(); }

#endif // TEST_DERIVATIVES_HH