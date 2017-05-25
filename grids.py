import numpy as np

from fields import Field


class Domain:
    """A representation of the domain on which to solve the Poisson equation"""

    def __init__(self, center, edges):
        self.center = tuple(center)
        self.edges = tuple(edges)

    def __eq__(self, other):
        return self.center == other.center and self.edges == other.edges

    def __ne__(self, other):
        return not self.__eq__(other)

class Grid:
    """A cartesian, equidistant grid in 2 or 3 dimensions"""

    def __init__(self, domain, shape):
        self.domain = domain
        self.shape = tuple(shape)

        if len(shape) == 2:
            nx, ny= shape
            self.boundary = [(ix, iy) for ix in range(nx)
                                for iy in range(ny)
                                if ix == 0 or ix == nx - 1 or
                                   iy == 0 or iy == ny - 1 ]
        elif len(shape) == 3:
            nx, ny, nz = shape
            self.boundary = [(ix, iy, iz) for ix in range(nx)
                                for iy in range(ny)
                                for iz in range(nz)
                                if ix == 0 or ix == nx - 1 or
                                   iy == 0 or iy == ny - 1 or
                                   iz == 0 or iz == nz - 1]
        else:
            raise Exception("Only 2 or 3 dimensions implemented")

        _edges = np.array(domain.edges)
        self._spacing =  _edges / (np.array(self.shape) - 1)
        self._origin = np.array(domain.center) - _edges / 2

    def __eq__(self, other):
        return self.shape == other.shape and self.domain == other.domain

    def __ne__(self, other):
        return not self.__eq__(other)

    def spacing(self):
        return tuple(self._spacing)

    def center(self):
        return self.domain.center

    def shift(self, translation_vector):
        self.domain.center += np.array(translation_vector)

    def loc(self, index_tuple):
        return self._origin + np.array(index_tuple) * self._spacing

    def array(self):
        return np.zeros(self.shape)

    def field(self):
        return Field(self)

    def field_from_function(self, func):
        field = Field(self)
        nx, ny, nz = self.shape
        for ind in ( (ix, iy, iz) for ix in range(nx)
                                  for iy in range(ny)
                                  for iz in range(nz)):
            x, y, z = self.loc(ind)
            field.values[ind] = func(x, y, z)
        return field

    def indices(self):
        nx, ny, nz = self.shape()
        return [(ix,iy,iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)]

    def is_on_boundary(self, index):

        for i in range(3):
            if index[i] == 0 or index[i] == self.shape[i]-1:
                return True
        return False


class MultiGrid:

    def __init__(self, root_grid):
        self.root = root_grid
        self.grids = [root_grid]
        self._build_sub_grids()

    def coarsify(self, field):

        from_level = self.level(field.grid)
        coarse_grid = self.grids[from_level+1]
        coarse_field = coarse_grid.field()

        nx, ny, nz = coarse_grid.shape
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    ixf, iyf, izf = 2*ix, 2*iy, 2*iz
                    coarse_field.values[ix, iy, iz] = field.values[ixf, iyf, izf]

        return coarse_field

    def has_coarser(self, grid):
        level = self.level(grid)
        return level < len(self.grids)-1

    def _bracket(self, index, max_index):

        if index % 2 == 0:
            return [index // 2]

        left = (index - 1) // 2
        if left >= 0:
            bracket = [left]
        right = (index + 1) // 2
        if right <= max_index:
            bracket.append(right)

        return bracket

    def _bracket_average(self, xbr, ybr, zbr, u):

        avg = 0.
        count = 0
        for ix in xbr:
            for iy in ybr:
                for iz in zbr:
                    avg += u[ix, iy, iz]
                    count += 1
        avg /= count
        return avg

    def refine(self, field):

        from_level = self.level(field.grid)
        fine_grid = self.grids[from_level-1]
        fine_field = fine_grid.field()

        nx, ny, nz = fine_grid.shape

        for ix in range(nx):
            xbracket = self._bracket(ix, nx-1)

            for iy in range(ny):
                ybracket = self._bracket(iy, ny - 1)

                for iz in range(nz):
                    zbracket = self._bracket(iz, nz - 1)

                    fine_field.values[ix, iy, iz] = self._bracket_average(xbracket, ybracket, zbracket, field.values)

        return fine_field

    def depth(self):
        return len(self.grids)

    def level(self, grid):
        for lvl, g in enumerate(self.grids):
            if g == grid:
                return lvl
        raise Exception("No such grid in multigrid")

    def grid(self, level):
        if level >= len(self.grids):
            raise IndexError
        return self.grids[level]

    def _build_sub_grids(self):

        grid = self.root

        while True:
            new_shape = tuple(np.array(grid.shape) // 2 + 1)

            new_grid = Grid(grid.domain, new_shape)
            self.grids.append(new_grid)

            if new_shape == (3, 3, 3):
                break

            for n in new_shape:
                if n < 3:
                    raise Exception("Invalid base grid")

            grid = new_grid