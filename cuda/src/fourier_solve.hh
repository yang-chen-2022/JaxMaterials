#ifndef FOURIER_SOLVE_HH
#define FOURIER_SOLVE_HH FOURIER_SOLVE_HH
#include "common.hh"

void initialize_xizero(float *dev_xi_zero_0,
                       float *dev_xi_zero_1,
                       float *dev_xi_zero_2,
                       const GridSpec grid_spec);

void fourier_solve(float *dev_sigma_hat, float *dev_r_hat,
                   const GridSpec grid_spec);

#endif // FOURIER_SOLVE_HH