/** @brief Implementation of profile.hh */
#include "profile.hh"
void profile_derivatives()
{
    int niter = 1000;
    GridSpec grid_spec;
    grid_spec.nx = 64;
    grid_spec.ny = 64;
    grid_spec.nz = 64;
    grid_spec.Lx = 1.0;
    grid_spec.Ly = 1.0;
    grid_spec.Lz = 1.0;

    int ncells = grid_spec.number_of_cells();

    // ==== device ====
    // allocate memory
    float *dev_u = nullptr;
    float *dev_du = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_u, ncells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_du, ncells * sizeof(float)));
    // measure time
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < niter; ++i)
    {
        backward_derivative_device(dev_u, dev_du, 0, grid_spec);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto t_finish = std::chrono::high_resolution_clock::now();
    double t_elapsed = double(std::chrono::duration_cast<std::chrono::microseconds>(t_finish - t_start).count()) / niter;
    printf("time per call [device] = %8.2f us \n", t_elapsed);
    // ==== host ====
    niter = 10;
    // allocate memory
    float *u = nullptr;
    float *du = nullptr;
    CUDA_CHECK(cudaMallocHost(&u, ncells * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&du, ncells * sizeof(float)));
    // measure time
    t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < niter; ++i)
    {
        backward_derivative_host(u, du, 0, grid_spec);
    }
    t_finish = std::chrono::high_resolution_clock::now();
    t_elapsed = double(std::chrono::duration_cast<std::chrono::microseconds>(t_finish - t_start).count()) / niter;
    printf("time per call [host]   = %8.2f us \n", t_elapsed);
    CUDA_CHECK(cudaFree(dev_u));
    CUDA_CHECK(cudaFree(dev_du));
    CUDA_CHECK(cudaFreeHost(u));
    CUDA_CHECK(cudaFreeHost(du));
}