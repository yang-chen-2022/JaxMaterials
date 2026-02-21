/** @brief Computation of derivatives on device and host */
#ifndef DERIVATIVES_HH
#define DERIVATIVES_HH DERIVATIVES_HH
#include "common.hh"
#include <cuda.h>

/** @brief Launch kernel to compute the backward derivative in x-direction
 *
 * du/dx_{a,b,c} = 1/(4 h_x) * ( u_{a,b,c}   + u_{a,b-1,c}   + u_{a,b,c-1}   +
 * u_{a,b-1,c-1}
 *                             + u_{a-1,b,c} + u_{a-1,b-1,c} + u_{a-1,b,c-1} +
 * u_{a-1,b-1,c-1} )
 *
 * @param[in] u: field for which the derivative is computed (device pointer)
 * @param[out] du_dx: resulting field du/dx (device pointer)
 * @param[in] grid_spec: grid specification
 */
void backward_derivative_x(float *u, float *du_dx, const GridSpec grid);

/** @brief Compute backward derivative in x-direction
 *
 * Host implementation for testing
 *
 * du/dx_{a,b,c} = 1/(4 h_x) * ( u_{a,b,c}   + u_{a,b-1,c}   + u_{a,b,c-1}   +
 * u_{a,b-1,c-1}
 *                             + u_{a-1,b,c} + u_{a-1,b-1,c} + u_{a-1,b,c-1} +
 * u_{a-1,b-1,c-1} )
 *
 * @param[in] u: field for which the derivative is computed (host pointer)
 * @param[out] du_dx: resulting field du/dx (host pointer)
 * @param[in] grid_spec: grid specification
 */
void backward_derivative_x_host(float *u, float *du_dx, const GridSpec grid);

#endif // DERIVATIVES_HH