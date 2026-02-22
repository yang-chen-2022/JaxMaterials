/** @brief Computation of derivatives on device and host */
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
 * @param[in] u: field for which the derivative is computed (device pointer)
 * @param[out] du: resulting field du/d{x,y,z} (device pointer)
 * @param[in] direction: coordinate direction in which the derivative is taken
 * @param[in] grid_spec: grid specification
 * @param[in] use_shared_memory use shared memory kernels?
 */
void backward_derivative_device(float *u, float *du,
                                const int direction,
                                const GridSpec grid_spec,
                                const bool use_shared_memory = false);

/** @brief Compute backward derivative in arbitrary direction
 *
 * Equivalent host implementation for testing
 *
 * @param[in] u: field for which the derivative is computed (host pointer)
 * @param[out] du: resulting field du/d{x,y,z} (host pointer)
 * @param[in] direction: coordinate direction in which the derivative is taken
 * @param[in] grid_spec: grid specification
 */
void backward_derivative_host(float *u, float *du,
                              const int direction,
                              const GridSpec grid_spec);

#endif // DERIVATIVES_HH