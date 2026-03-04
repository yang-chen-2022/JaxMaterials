/** @brief Computation of derivatives on device and host
 *
 * Provides methods for computing discretised derivatives in real space
 */
#ifndef DERIVATIVES_HH
#define DERIVATIVES_HH DERIVATIVES_HH
#include <stdexcept>
#include "common.hh"
#include <cuda.h>

/** @brief Launch kernel to compute the backward derivative in arbitrary direction
 *
 * du/dx_{a,b,c} = 1/(4 h_x) * ( u_{a,b,c}     + u_{a,b-1,c}
 *                             + u_{a,b,c-1}   + u_{a,b-1,c-1}
 *                             - u_{a-1,b,c}   - u_{a-1,b-1,c}
 *                             - u_{a-1,b,c-1} - u_{a-1,b-1,c-1} )
 *
 * du/dy_{a,b,c} = 1/(4 h_y) * ( u_{a,b,c}     + u_{a-1,b,c}
 *                             + u_{a,b,c-1}   + u_{a-1,b,c-1}
 *                             - u_{a,b-1,c}   - u_{a-1,b-1,c}
 *                             - u_{a,b-1,c-1} - u_{a-1,b-1,c-1} )
 *
 * du/dz_{a,b,c} = 1/(4 h_z) * ( u_{a,b,c}     + u_{a-1,b,c}
 *                             + u_{a,b-1,c}   + u_{a-1,b-1,c}
 *                             - u_{a,b,c-1}   - u_{a-1,b,c-1}
 *                             - u_{a,b-1,c-1} - u_{a-1,b-1,c-1} )
 *
 * @param[in] u field for which the derivative is computed (device pointer)
 * @param[out] du resulting field du/d{x,y,z} (device pointer)
 * @param[in] direction coordinate direction in which the derivative is taken
 * @param[in] grid_spec grid specification
 * @param[in] increment increment values instead of overwriting them
 */
void backward_derivative_device(float *__restrict__ u,
                                float *__restrict__ du,
                                const int direction,
                                const GridSpec grid_spec,
                                const bool increment = false);

/** @brief Compute backward derivative in arbitrary direction
 *
 * Equivalent host implementation for testing
 *
 * @param[in] u: field for which the derivative is computed (host pointer)
 * @param[out] du: resulting field du/d{x,y,z} (host pointer)
 * @param[in] direction: coordinate direction in which the derivative is taken
 * @param[in] increment increment values instead of overwriting them
 * @param[in] grid_spec: grid specification
 */
void backward_derivative_host(float *u, float *du,
                              const int direction,
                              const GridSpec grid_spec,
                              const bool increment = false);

/** @brief Compute backward divergence of symmetric tensor on device
 *
 * For a given symmetric tensor sigma_{ij} in Voigt notation, compute the divergence
 * div(sigma)_i = dsigma_{ij}/dx_j using backward derivatives.
 *
 * @param[in] dev_sigma: field for which the derivative is computed (device pointer, size 6*nvoxels)
 * @param[out] dev_div_sigma: resulting divergence (device pointer, size 3*nvoxels)
 * @param[in] grid_spec: grid specification
 */
void backward_divergence_device(float *__restrict__ dev_sigma,
                                float *__restrict__ dev_div_sigma,
                                const GridSpec grid_spec);

/** @brief Compute backward divergence of symmetric tensor on host
 *
 * Equivalent implementation on host
 *
 * @param[in] sigma: field for which the derivative is computed (host pointer, size 6*nvoxels)
 * @param[out] div_sigma: resulting divergence (host pointer, size 3*nvoxels)
 * @param[in] grid_spec: grid specification
 */
void backward_divergence_host(float *sigma, float *div_sigma,
                              const GridSpec grid_spec);

#endif // DERIVATIVES_HH