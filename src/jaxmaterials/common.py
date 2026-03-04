"""Common definitions"""

__all__ = "GridSpec"


class GridSpec:
    """Simple class for describing a structured grid"""

    def __init__(self, nx, ny, nz, Lx, Ly, Lz):
        """Initialise instance

        :arg nx: number of voxels in x-direction
        :arg ny: number of voxels in y-direction
        :arg nz: number of voxels in z-direction
        :arg Lx: domain size in x-direction
        :arg Ly: domain size in y-direction
        :arg Lz: domain size in z-direction
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

    @property
    def number_of_voxels(self):
        """Return total number of voxels"""
        return self.nx * self.ny * self.nz
