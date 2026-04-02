import numpy as np
from contextlib import contextmanager
import time

from jax import numpy as jnp

__all__ = ["measure_time", "save_to_vtk"]


@contextmanager
def measure_time(label):
    """Measure the time it takes to execute a block of code

    :arg label: label for the time measurement
    """
    t_start = time.perf_counter()
    try:
        yield
    finally:
        t_finish = time.perf_counter()
        t_elapsed = t_finish - t_start
        print(f"time [{label}] = {t_elapsed:8.2f} s")


def save_to_vtk(data, grid_spec, filename, location="centre"):
    """Save fields to VTK file

    :arg data: dictionary of the form {"label":field} where field is an array of
               shape (nx,ny,nz)
    :arg grid_spec: Specification of computational grid
    :arg filename: name of file to save to
    :arg location: location of data within voxel. Currently only "centre" is supported
    """
    assert location == "centre"
    shape = next(iter(data.values())).shape
    nx, ny, nz = shape
    with open(filename, mode="w", encoding="utf8") as f:

        print("# vtk DataFile Version 2.0", file=f)
        print("data", file=f)
        print("ASCII", file=f)
        print("DATASET RECTILINEAR_GRID", file=f)
        print(f"DIMENSIONS {nx+1} {ny+1} {nz+1}", file=f)
        for n, extent, dim_label in zip(
            shape, (grid_spec.Lx, grid_spec.Ly, grid_spec.Lz), "XYZ"
        ):
            print(f"{dim_label}_COORDINATES {n+1} float", file=f)
            print(
                " ".join([f"{x:12.8f}" for x in np.linspace(0, extent, num=n + 1)]),
                file=f,
            )
        print("", file=f)
        print(f"CELL_DATA {nx*ny*nz}", file=f)
        for key, value in data.items():
            print(f"SCALARS {key} float 1", file=f)
            print("LOOKUP_TABLE default", file=f)
            print(
                "\n".join([f"{v:12.8f}" for v in value.flatten(order="F")]),
                file=f,
            )




def voigt_to_tensor(v):
    """Convert 6-component Voigt notation (6, Nx, Ny, Nz) to 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3)."""
    v0, v1, v2, v3, v4, v5 = v[0], v[1], v[2], v[3], v[4], v[5]
    
    row0 = jnp.stack([v0, v5, v4], axis=-1) #slots into the end
    row1 = jnp.stack([v5, v1, v3], axis=-1) 
    row2 = jnp.stack([v4, v3, v2], axis=-1)
    
    return jnp.stack([row0, row1, row2], axis=-2) #


def tensor_to_voigt(t):
    """Convert 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3) back to 6-component Voigt (6, Nx, Ny, Nz)."""
    v0 = t[..., 0, 0]
    v1 = t[..., 1, 1]
    v2 = t[..., 2, 2]
    v3 = t[..., 1, 2]
    v4 = t[..., 0, 2]
    v5 = t[..., 0, 1]
    
    return jnp.stack([v0, v1, v2, v3, v4, v5], axis=0)
