import numpy as np
import torch
from jaxmaterials.distributions_fibres import FibreRadiusDistribution
from jaxmaterials.data_generator import LayerFibresDataset
from jaxmaterials.utilities import save_to_vtk

domain_size = [0.5, 0.3, 0.2]
d_void = 0.01
number_of_cells = [5 * 32, 3 * 32, 2 * 32]

r_avg = 7.5e-3
r_min = 5.0e-3
r_max = 10.0e-3
sigma = 0.5e-3
volume_fraction = 0.3
radius_distribution = FibreRadiusDistribution(
    r_avg=r_avg, r_min=r_min, r_max=r_max, sigma=sigma, gaussian=True
)
nlayers = 3
n_samples = 8
batch_size = 4
rng = np.random.default_rng(seed=5713853)

dataset = LayerFibresDataset(
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
)

for k in range(n_samples):
    layer_boundaries, fibre_positions, fibre_radii, fibre_orientations = (
        dataset.generate_fibre_positions()
    )
    dataset.visualise(
        layer_boundaries,
        fibre_positions,
        fibre_radii,
        fibre_orientations,
        filename=f"fibres_{k+1:03d}.pdf",
    )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for data, sigma_bar in iter(dataloader):
    print(data.shape, sigma_bar.shape)

for k in range(4):
    data, sigma_bar = dataset[k]
    save_to_vtk(
        {"mu": data[0], "lambda": data[1]},
        domain_size,
        filename=f"data_{k+1:03d}.vtk",
    )
