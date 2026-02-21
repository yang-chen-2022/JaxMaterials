#include "common.hh"
#include "lippmann_schwinger.hh"

int main() {
  // domain size
  GridSpec grid_spec;
  grid_spec.nx = 64;
  grid_spec.ny = 64;
  grid_spec.nz = 64;
  grid_spec.Lx = 1.0;
  grid_spec.Ly = 1.0;
  grid_spec.Lz = 1.0;
  lippmann_schwinger_solve(grid_spec);
  return 0;
}