"""Main script for visualising layered fibre dataset"""

import numpy as np
from jaxmaterials.distributions_fibres import FibreRadiusDistribution
from jaxmaterials.data import (
    LayeredFibresDataset,
    LayeredFibresDatasetGenerator,
    visualise_fibres,
)
from jaxmaterials.utilities import save_to_vtk

# Domain
domain_size = [0.5, 0.3, 0.2]
d_void = 0.01
N = 32
number_of_cells = [5 * N, 3 * N, 2 * N]

# Distribution of fibre radii
r_avg = 7.5e-3
r_min = 5.0e-3
r_max = 10.0e-3
sigma = 0.5e-3
volume_fraction = 0.3
radius_distribution = FibreRadiusDistribution(
    r_avg=r_avg, r_min=r_min, r_max=r_max, sigma=sigma, gaussian=True
)
# number of layers with fibres
nlayers = 3

n_samples = 8
rng = np.random.default_rng(seed=5713853)

dataset_generator = LayeredFibresDatasetGenerator(
    domain_size,
    number_of_cells,
    nlayers,
    d_void,
    n_samples,
    radius_distribution,
    volume_fraction=volume_fraction,
    mu_fibre=0.5,
    mu_material=1.0,
    mu_void=0.1,
    lambda_fibre=0.1,
    lambda_material=0.5,
    lambda_void=1.0,
    dtype=np.float64,
    rng=rng,
    verbose=True,
)

# Visualise the projected fibre locations for samples
for k in range(n_samples):
    layer_boundaries, fibre_positions, fibre_radii, fibre_orientations = (
        dataset_generator.generate_fibre_positions()
    )
    visualise_fibres(
        domain_size,
        layer_boundaries,
        fibre_positions,
        fibre_radii,
        fibre_orientations,
        filename=f"fibres_{k+1:03d}.pdf",
    )

filename = "fibres.h5"
dataset_generator.generate()
dataset_generator.save_hdf5(filename)
dataset = LayeredFibresDataset(filename)


# Save gridded material properties to vtk files
for k in range(n_samples):
    data, sigma_bar = dataset[k]
    save_to_vtk(
        {"mu": data[0], "lambda": data[1]},
        domain_size,
        filename=f"data_{k+1:03d}.vtk",
    )
