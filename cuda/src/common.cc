/* Implementation of common.hh */
#include "common.hh"

/* Initialise field */
void init_field(float *u, const GridSpec grid_spec)
{
  int nx = grid_spec.nx;
  int ny = grid_spec.ny;
  int nz = grid_spec.nz;
  for (int i = 0; i < nx; ++i)
  {
    for (int j = 0; j < ny; ++j)
    {
      for (int k = 0; k < nz; ++k)
      {
        u[IDX(nx, ny, nz, i, j, k)] = rand() / (float(RAND_MAX));
      }
    }
  }
}

/* Compite relative difference between two fields  */
float relative_difference(float *u, float *u_ref, const GridSpec grid_spec)
{
  size_t n = grid_spec.number_of_cells();
  float nrm2 = 0;
  float diff_nrm2 = 0;
  for (int ell = 0; ell < n; ++ell)
  {
    float _u = u_ref[ell];
    float _du = u[ell] - u_ref[ell];
    nrm2 += _u * _u;
    diff_nrm2 += _du * _du;
  }
  return sqrt(diff_nrm2 / nrm2);
}
