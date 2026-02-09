import numpy as np


def save_to_vtk(data, domain_size, filename, location="centre"):
    shape = next(iter(data.values())).shape
    nx, ny, nz = shape
    with open(filename, mode="w", encoding="utf8") as f:

        print("# vtk DataFile Version 2.0", file=f)
        print("data", file=f)
        print("ASCII", file=f)
        print("DATASET RECTILINEAR_GRID", file=f)
        print(f"DIMENSIONS {nx+1} {ny+1} {nz+1}", file=f)
        for n, extent, dim_label in zip(shape, domain_size, "XYZ"):
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
