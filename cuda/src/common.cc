/** @brief Implementation of common.hh */
#include "common.hh"

/* Compite relative difference between two fields  */
float relative_difference(float *u, float *u_ref, const int ndof) {
  float nrm2 = 0;
  float diff_nrm2 = 0;
  for (int ell = 0; ell < ndof; ++ell) {
    float _u = u_ref[ell];
    float _du = u[ell] - u_ref[ell];
    nrm2 += _u * _u;
    diff_nrm2 += _du * _du;
  }
  return sqrt(diff_nrm2 / nrm2);
}
