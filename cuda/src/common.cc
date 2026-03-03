/** @brief Implementation of common.hh */
#include "common.hh"

/* Compite relative difference between two fields  */
float relative_difference(float *u, float *u_ref, const size_t ndof) {
  float nrm2 = 0;
  float diff_nrm2 = 0;
  for (size_t ell = 0; ell < ndof; ++ell) {
    float _u = u_ref[ell];
    float _du = u[ell] - u_ref[ell];
    nrm2 += _u * _u;
    diff_nrm2 += _du * _du;
  }
  return sqrt(diff_nrm2 / nrm2);
}

/* Compute norm of vector field */
float vector_norm(float *u, const size_t nvoxels) {
  float nrm2 = 0;
  for (size_t ell = 0; ell < 3 * nvoxels; ++ell) {
    nrm2 += u[ell] * u[ell];
  }
  return sqrt(nrm2);
}

/* Compute norm of vector field */
float vector_norm(cufftComplex *u, const size_t nvoxels) {
  float nrm2 = 0;
  for (size_t ell = 0; ell < 3 * nvoxels; ++ell) {
    nrm2 += u[ell].x * u[ell].x + u[ell].y * u[ell].y;
  }
  return sqrt(nrm2);
}

/* Compute norm of tensor field */
float tensor_norm(float *tau, const size_t nvoxels) {
  float nrm2 = 0;
  for (size_t ell = 0; ell < nvoxels; ++ell) {
    for (size_t alpha = 0; alpha < 3; ++alpha)
      nrm2 += tau[alpha * nvoxels + ell] * tau[alpha * nvoxels + ell];
    for (size_t alpha = 3; alpha < 6; ++alpha)
      nrm2 += 2 * tau[alpha * nvoxels + ell] * tau[alpha * nvoxels + ell];
  }
  return sqrt(nrm2);
}

/* Compute norm of tensor field */
float tensor_norm(cufftComplex *tau, const size_t nvoxels) {
  float nrm2 = 0;
  for (int ell = 0; ell < nvoxels; ++ell) {
    for (int alpha = 0; alpha < 3; ++alpha)
      nrm2 += tau[alpha * nvoxels + ell].x * tau[alpha * nvoxels + ell].x +
              tau[alpha * nvoxels + ell].y * tau[alpha * nvoxels + ell].y;
    for (int alpha = 3; alpha < 6; ++alpha)
      nrm2 += 2 * (tau[alpha * nvoxels + ell].x * tau[alpha * nvoxels + ell].x +
                   tau[alpha * nvoxels + ell].y * tau[alpha * nvoxels + ell].y);
  }
  return sqrt(nrm2);
}
