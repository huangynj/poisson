import unittest
from numpy.testing import assert_almost_equal

from grids import *


class GridTest(unittest.TestCase):

    def setUp(self):
        self.domain = Domain(center=(0,0,0), edges=(1,1,1))
        self.grid = Grid(self.domain, shape=(11,10,9))

    def test_make_boundary(self):
        domain = Domain(center=(0,0,0), edges=(1,1,1))
        grid = Grid(domain, shape=(11,11,11))
        interior_points = 9*9*9
        boundary_points = 11*11*11 - interior_points
        self.assertEqual(boundary_points, len(grid.boundary))

    def test_shape(self):
        domain = Domain(center=(0,0,0), edges=(1,1,1))
        grid = Grid(domain, shape=(11,10,9))
        nx, ny, nz = grid.shape
        self.assertEqual(11, nx)
        self.assertEqual(10, ny)
        self.assertEqual(9, nz)

    def test_spacing(self):
        assert_almost_equal((1/10, 1/9, 1/8), self.grid.spacing())

    def test_center(self):
        assert_almost_equal(np.array((0,0,0)), self.grid.center())
        assert_almost_equal(((0,0,0)), self.grid.center())
        assert_almost_equal((-0.5, -0.5, -0.5), self.grid._origin )

    def test_shift(self):
        self.grid.shift((1,2,3))
        assert_almost_equal(np.array((1,2,3)), self.grid.center())
        assert_almost_equal(np.array((1,2,3)), self.grid.domain.center)

    def test_location(self):
        x, y, z = self.grid.loc( (0, 9, 4))
        self.assertAlmostEqual(-0.5, x)
        self.assertAlmostEqual(0.5, y)
        self.assertAlmostEqual(0, z)

    def test_field_from_function(self):

        def f(x,y,z):
            return x**2 + y**2 + z**2

        field = self.grid.field_from_function(f)
        a = (1/2)**2*3
        self.assertAlmostEqual(a, field.values[0,0,0] )

    def test_grid_equality(self):

        domain = Domain(center=(0,0,0), edges=(1,1,1))
        grid1 = Grid(domain, shape=(3,4,5))
        grid2 = Grid(domain, shape=(3,4,5))
        grid3 = Grid(domain, shape=(3, 4, 6))

        self.assertTrue(id(grid1) != id(grid2))
        self.assertTrue(grid1 == grid2)
        self.assertTrue(grid1 != grid3)


class MultiGridTest(unittest.TestCase):

    def test_multigrid_setup(self):
        root = Grid(Domain(center=(0,0,0), edges=(1,1,1)), (17,17,17))
        multigrid = MultiGrid(root)

        self.assertEqual(4, multigrid.depth())

    def test_level_for_grid(self):
        root = Grid(Domain(center=(0,0,0), edges=(1,1,1)), (17,17,17))
        multigrid = MultiGrid(root)

        grid = Grid(Domain(center=(0,0,0), edges=(1,1,1)), (5,5,5))
        level = multigrid.level(grid)
        self.assertEqual(2, level)

    def test_coarsify(self):
        root = Grid(Domain(center=(0, 0, 0), edges=(1, 1, 1)), (17, 17, 17))
        multigrid = MultiGrid(root)

        grid = Grid(Domain(center=(0, 0, 0), edges=(1, 1, 1)), (9, 9, 9))

        def f(x, y, z):
            return x+y+z

        field = grid.field_from_function(f)
        coarser_field = multigrid.coarsify(field)

        level_fine = multigrid.level(grid)
        self.assertEqual(1, level_fine)

        level_coarse = multigrid.level(coarser_field.grid)
        self.assertEqual(2, level_coarse)

        value_fine = field.values[(2, 2, 2)]
        value_coarse = coarser_field.values[1,1,1]
        self.assertTrue(value_fine != 0.)
        self.assertAlmostEqual(value_fine, value_coarse)

    def test_refine(self):

        root = Grid(Domain(center=(0, 0, 0), edges=(1, 1, 1)), (17, 17, 17))
        multigrid = MultiGrid(root)

        grid = Grid(Domain(center=(0, 0, 0), edges=(1, 1, 1)), (5, 5, 5))

        def f(x, y, z):
            return x+y+z

        coarse_field = grid.field_from_function(f)

        fine_field = multigrid.refine(coarse_field)

        level_coarse = multigrid.level(grid)
        self.assertEqual(2, level_coarse)

        level_fine = multigrid.level(fine_field.grid)
        self.assertEqual(1, level_fine)

        self.assertTrue(abs(coarse_field.values[1, 1, 1]) > 0)
        self.assertAlmostEqual(coarse_field.values[1,1,1], fine_field.values[2,2,2])
        self.assertAlmostEqual(0.5*(coarse_field.values[1,1,1]+coarse_field.values[1,2,1])
                               , fine_field.values[2,3,2])


        self.assertAlmostEqual(0.25*(coarse_field.values[0,0,0]+coarse_field.values[0,1,0]
                                    +coarse_field.values[1, 0, 0] + coarse_field.values[1, 1, 0]
                                    ),fine_field.values[1,1,0])

if __name__ == '__main__':
    unittest.main()