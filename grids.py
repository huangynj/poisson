""" A utility module with classes for grid representations """

import numpy as np

class Domain:
    """A representation of the domain on which to solve the Poisson equation"""

    def __init__(self, center, edges):
        """ Constructor for Domain
        
        :param center: The center of the domain. Must be 3D list, array or tuple.  
        :param edges:  The lenghts of the edges of the domain. Must be 3D list, array or tuple.
        """
        self.center = tuple(center)
        self.edges = tuple(edges)

    def __eq__(self, other):
        return self.center == other.center and self.edges == other.edges

    def __ne__(self, other):
        return not self.__eq__(other)

class Grid:
    """A cartesian, equidistant grid in 3 dimensions"""

    def __init__(self, domain, shape):
        """ Constructor for Grid
        
        :param domain: The region of space the grid shall cover. Must be Domain.
        :param shape: The number of grid points in each direction. Must be 3D tuple.
        """

        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain instance")
        if not isinstance(shape, tuple) or len(shape) != 3:
            raise TypeError("shape must be a 3d tuple")

        self.domain = domain
        self.shape = tuple(shape)

        nx, ny, nz = shape
        self.boundary = [(ix, iy, iz) for ix in range(nx)
                            for iy in range(ny)
                            for iz in range(nz)
                            if ix == 0 or ix == nx - 1 or
                               iy == 0 or iy == ny - 1 or
                               iz == 0 or iz == nz - 1]

        _edges = np.array(domain.edges)
        self._spacing =  _edges / (np.array(self.shape) - 1)
        self._origin = np.array(domain.center) - _edges / 2

    def __eq__(self, other):
        return self.shape == other.shape and self.domain == other.domain

    def __ne__(self, other):
        return not self.__eq__(other)

    def spacing(self):
        """ The spacings between grid points of the equidistant grid.
        
        :return: (dx, dy, dz) as tuple 
        """
        return tuple(self._spacing)

    def center(self):
        """ The center of the domain of the grid.
        
        :return: the 3D tuple representing the center of the domain.
        """
        return self.domain.center

    def shift(self, translation_vector):
        """ Shift the grid (and domain) by the given translation vector
        
        :param translation_vector: a 3D list, array or tuple
        """
        self.domain.center += tuple(np.array(translation_vector))

    def loc(self, index_tuple):
        """ Returns the (x, y, z) coordinates for the grid point of the index_tuple
        
        :param index_tuple: a 3D list, array or tuple
        :return: a tuple with the (x, y, z) coordinates of the grid point
        """
        return self._origin + np.array(index_tuple) * self._spacing

    def array(self):
        """ Factory function to create a zero numpy array with suitable shape for the grid
        
        :return: a zero numpy array compatible with the grid
        """
        return np.zeros(self.shape)

    def field(self):
        """ Factory function to create a zero Field with suitable shape for the grid
        
        :return: a zero Field based on the grid
        """
        return Field(self)

    def field_from_function(self, func):
        """ Factory function to create a Field with suitable shape for the grid. The values
            are computed by a user-defined function func(x, y, z)
        
        :param func: a function f(x, y, z) -> real number
        :return: a new Field filled with values calculated by func
        """
        field = Field(self)
        nx, ny, nz = self.shape
        for ind in ( (ix, iy, iz) for ix in range(nx)
                                  for iy in range(ny)
                                  for iz in range(nz)):
            x, y, z = self.loc(ind)
            field.values[ind] = func(x, y, z)
        return field

    def indices(self):
        """ Returns a list of all index tuples of the grid
        
        :return: list of index tuples
        """
        nx, ny, nz = self.shape()
        return [(ix,iy,iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)]

    def is_on_boundary(self, index):
        """ Tells whether grid point 'index' is on boundary or not
        
        :param index: 
        :return: boolean
        """
        for i in range(3):
            if index[i] == 0 or index[i] == self.shape[i]-1:
                return True
        return False


class MultiGrid:
    """ A represenation of a multigrid, i.e. a set of grids with different coarseness.
        Can translate fields between grids of different coarseness.
    """
    
    def __init__(self, root_grid):
        """ Constructor for MultiGrid
        
        :param root_grid: the finest grid of the MultiGrid to construct
        """
        self.root = root_grid
        self.grids = [root_grid]
        self._build_sub_grids()

    def coarsify(self, field):
        """ Translate the field to the next coarser grid
        
        :param field: Field instance on a fine grid to translate 
        :return: new, translated Field instance on coarser grid
        """

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
        """ Tells if there is a grid coarser than 'grid' in the multigrid 
        
        :param grid: The Grid instance to compare
        :return: boolean
        """
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
        """ Translate the field to the next finer grid
        
        :param field: the Field instance on a coarse grid to translate 
        :return: the new, translated Field instance on the finer grid
        """

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
        """ Returns the number of grids in the multigrid
        
        :return: number of grids
        """
        return len(self.grids)

    def level(self, grid):
        """ Returns the level of 'grid' in the multigrid. The finest grid is level 0, next coarser is 1, ...
        
        :param grid: The Grid instance for which to get the level
        :return: the level number
        """
        for lvl, g in enumerate(self.grids):
            if g == grid:
                return lvl
        raise Exception("No such grid in multigrid")

    def grid(self, level):
        """ Returns the Grid instance for a given level in the multigrid
        
        :param level: int
        :return: Grid instance 
        """
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


class Field:
    """ A Field is the set of function values together with its grid.    
    """

    def __init__(self, grid):
        self.grid = grid
        self.values = np.zeros(grid.shape)

    def __getitem__(self, index):
        """ []-Accessor. Returns the a tuple with the location on the grid and the value stored there.
        
        :param index: 
        :return: ( (x,y,z) , value ) all floats
        """
        return (self.grid.loc(index), self.values[index])
