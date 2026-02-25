#ifndef LIPPMANN_SCHWINGER_HH
#define LIPPMANN_SCHWINGER_HH LIPPMANN_SCHWINGER_HH

#include <algorithm>
#include "cufft.h"
#include "cublas_v2.h"
#include "common.hh"
#include "derivatives.hh"
#include "fourier_solve.hh"

/** @brief Compute stress on device
 *
 * Given epsilon and spatially varying Lame parameters lambda, mu compute the
 * stress sigma according to sigma_{ij} = C_{ijkl} epsilon_{kl} where
 *
 *      C_{ijkl} = lambda*delta_{ij}delta_{kl} + mu*(delta_{ik}delta_{jl}+delta_{il}delta_{jk})
 *
 * @param[in] dev_epsilon strain epsilon in real space (device array, size 6*ncells)
 * @param[out] dev_sigma resulting stress sigma in real space (device array, size 6*ncells)
 * @param[in] dev_lambda Lame parameter lambda (device array, size ncells)
 * @param[in] dev_mu Lame parameter mu (device array, size ncells)
 * @param[in] grid_spec Specification of computational grid
 */
void compute_stress(float *dev_epsilon, float *dev_sigma,
                    float *dev_lambda, float *dev_mu,
                    int *cells,
                    float *extents);

/** @brief Increment solution
 *
 * Increment
 *
 *      epsilon -> epsilon + 1/ncells * r
 *
 * where the factor 1/ncells arises since the inverse Fourier transformation
 * in cuFFT is not normalised.
 *
 * @param[inout] dev_epsilon solution (device array, size 6*ncells)
 * @param[in] dev_r increment (device array, size 6*ncells)
 * @param[in] grid_spec specification of computational grid
 */
void increment_solution(float *dev_epsilon, cufftComplex *dev_r,
                        const GridSpec grid_spec);

/** @brief Compute normalised divergence for stopping criterion in Fourier space
 *
 * Compute the relative divergence norm
 *
 *      sqrt(<||div(sigma)||^2>) / ||<sigma>||
 *
 * Which in Fourier space is given by
 *
 *      sqrt(<||xi.hat(sigma)||^2>) / ||hat(sigma)(0)||
 *
 * @param[in] dev_sigma_hat stress in Fourier space
 * @param[in] dev_div_sigma_hat divergence of stress in Fourier space
 * @param[in] handle cuBLAS handle
 * @param[in] grid_spec specification of grid
 */
float relative_divergence_norm(cufftComplex *dev_sigma_hat, cufftComplex *dev_div_sigma_hat,
                               cublasHandle_t &handle, const GridSpec grid_spec);

/** @brief Solve linear elasticity problem with Lippmann-Schwinger iteration
 *
 * @param[in] lambda Lame parameter lambda (host array, size ncells)
 * @param[in] mu Lame parameter mu (host array, size ncells)
 * @param[in] epsilon_bar average value of epsilon (host array, size 6)
 * @param[out] epsilon Resulting strain (host array, size 6*ncells)
 * @param[out] sigma Resulting stress (host array, size 6*ncells)
 * @param[in] cells Number of cells (nx,ny,nz)
 * @param[in] extents Size of domain in each direction (Lx,Ly,Lz)
 */
extern "C"
{
    void lippmann_schwinger_solve(float *lambda, float *mu, float *epsilon_bar,
                                  float *epsilon, float *sigma,
                                  int *cells,
                                  float *extents);
}

#endif // LIPPMANN_SCHWINGER_HH