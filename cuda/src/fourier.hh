#ifndef FOURIER_HH
#define FOURIER_HH FOURIER_HH
#include "cufft.h"
#include "common.hh"

/** @brief Construct Fourier vectors tilde{xi}_j on device
 *
 * The vectors are defined as
 *
 *    tilde(xi)_0 = 2/h_0 sin(xi_0/2) cos(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_1 = 2/h_1 cos(xi_0/2) sin(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_2 = 2/h_2 cos(xi_0/2) cos(xi_1/2) sin(xi_2/2)
 *
 * with (xi_d)_j = 2 pi k_j / N_d and k_j = 0,1,...,N_d-1
 *
 *
 * @param[out] dev_xi vectors tilde{xi} (device array, size 3*nvoxels)
 * @param[in] grid_spec specification of grid
 */
void initialize_xi(float *dev_xi, const GridSpec grid_spec);

/** @brief Construct Fourier vectors ring{tilde{xi}}_j on device
 *
 * The vectors are defined as
 *
 *    tilde(xi)_0 = 2/h_0 sin(xi_0/2) cos(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_1 = 2/h_1 cos(xi_0/2) sin(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_2 = 2/h_2 cos(xi_0/2) cos(xi_1/2) sin(xi_2/2)
 *
 * with (xi_d)_j = 2 pi k_j / N_d and k_j = 0,1,...,N_d-1
 *
 * We then set
 *
 *    ring{tilde{xi}}_j = tilde{xi}_j / ||tilde{xi}_j|| if ||tilde{xi}_j|| != 0
 *
 * and ring{tilde{xi}}_j = 0 if ||tilde{xi}_j|| = 0
 *
 * @param[out] dev_xi_zero vectors ring{tilde{xi}} (device array, size 3*nvoxels)
 * @param[in] grid_spec specification of grid
 */
void initialize_xizero(float *dev_xi_zero, const GridSpec grid_spec);

/** @brief Construct Fourier vectors ring{tilde{xi}}_j on host
 *
 * Equivalent host implementation of initialize_xizero()
 *
 * @param[out] dev_xi_zero vectors ring{tilde{xi}} (host array, size 3*nvoxels)
 * @param[in] grid_spec specification of grid
 */
void initialize_xizero_host(float *xi_zero, const GridSpec grid_spec);

/** @brief Compute divergence in Fourier space
 *
 * Compute xi.sigma
 *
 * @param[in] dev_sigma_hat stress in Fourier space (device array, size 6*Nvoxels)
 * @param[out] dev_div_sigma_hat resulting divergence xi.sigma (device array, size 3*Nvoxels)
 * @param[in] dev_xi Fourier vectors (device array, size 3*Nvoxels)
 * @param[in] grid_spec specification of computational grid
 */
void divergence_fourier(cufftComplex *dev_sigma_hat,
                        cufftComplex *dev_div_sigma_hat,
                        float *dev_xi,
                        const GridSpec grid_spec);

/** @brief Solve elasticity equation for homogeneous isotropic reference
 * material
 *
 * Compute hat{epsilon} = -Gamma^0 hat{tau}
 *
 * @param[in] dev_tau_hat right hand side hat{tau} (device array, size 6*nvoxels)
 * @param[out] dev_epsilon_hat resulting hat{epsilon} (device array, size
 * 6*nvoxels)
 * @param[in] dev_xi_zero Fourier vectors xi_zero (device array, size 3*n),
 *                        as computed by initialize_xizero()
 * @param[in] lambda_0 Lame parameter lambda_0 of reference material
 * @param[in] mu_0 Lame parameter mu_0 of reference material
 * @param[in] grid_spec Grid specification
 */
void fourier_solve_device(cufftComplex *dev_tau_hat, cufftComplex *dev_epsilon_hat,
                          float *dev_xi_zero, const float lambda_0,
                          const float mu_0, const GridSpec grid_spec);

#endif // FOURIER_HH