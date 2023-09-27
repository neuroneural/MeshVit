class FixedCoordGenerator:
    def __init__(self, volume_dims, subvolume_size):
        """
        volume_dims: Tuple representing the size of the whole volume. Format: (z_dim, y_dim, x_dim)
        subvolume_size: Integer representing the edge size of a cubic subvolume
        """
        self.volume_dims = (volume_dims,volume_dims,volume_dims)
        self.subvolume_size = subvolume_size
        self.coords_list = self._generate_fixed_coords()

    def _generate_fixed_coords(self):
        z_dim, y_dim, x_dim = self.volume_dims
        s = self.subvolume_size
        coords_list = []

        # Ensuring that the full volume can be divided into equal subvolumes without remainders
        assert z_dim % s == 0, "Z dimension of volume is not divisible by subvolume size"
        assert y_dim % s == 0, "Y dimension of volume is not divisible by subvolume size"
        assert x_dim % s == 0, "X dimension of volume is not divisible by subvolume size"

        for z in range(0, z_dim, s):
            for y in range(0, y_dim, s):
                for x in range(0, x_dim, s):
                    coords_list.append(((z, z+s), (y, y+s), (x, x+s)))

        return coords_list

    def get_coordinates(self, mode = None):
        return self.coords_list
