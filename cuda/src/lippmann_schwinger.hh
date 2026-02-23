#ifndef LIPPMANN_SCHWINGER_HH
#define LIPPMANN_SCHWINGER_HH LIPPMANN_SCHWINGER_HH

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
 * @param[in] dev_lambda Lame parameter lambda (device array, size 6*ncells)
 * @param[in] dev_mu Lame parameter mu (device array, size 6*ncells)
 * @param[in] grid_spec Specification of computational grid
 */
void compute_stress(float *dev_epsilon, float *dev_sigma,
                    float *dev_lambda, float *dev_mu,
                    const GridSpec grid_spec);

void lippmann_schwinger_solve(const GridSpec grid_spec);

#endif // LIPPMANN_SCHWINGER_HH