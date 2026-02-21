#ifndef COMMON_HH
#define COMMON_HH COMMON_HH

#include <math.h>
#include <stdio.h>

// Block size used for kernel launches
#define BLOCKSIZE_X 8
#define BLOCKSIZE_Y 8
#define BLOCKSIZE_Z 8

// Convert 3d index to linear index
#define IDX(Nx, Ny, Nz, i, j, k) ((i) + (Nx) * ((j) + (Ny) * (k)))

/* Specification of grid*/
struct GridSpec
{
    int nx;   // Number of grid cells in x-direction
    int ny;   // Number of grid cells in y-direction
    int nz;   // Number of grid cells in z-direction
    float Lx; // Size of domain in x-direction
    float Ly; // Size of domain in y-direction
    float Lz; // Size of domain in z-direction
    int number_of_cells() const { return nx * ny * nz; }
};

/** @brief initialise field
 *
 * Set value of fields to random values for testing
 *
 * @param[out] u: field to set (host pointer)
 * @param[in] grid_spec: Grid specification
 */
void init_field(float *u, const GridSpec grid_spec);

/** @brief Compute relative L2 norm
 *
 * Computes ||u-u_{ref}||_2 / ||u_{ref}||_2
 *
 * @param[out] u: field (host pointer)
 * @param[out] u_ref: reference field to compare to (host pointer)
 * @param[in] grid_spec: Grid specification
 */
float relative_difference(float *u, float *u_ref, const GridSpec grid_spec);

#endif // COMMON_HH
