import math
import unittest

from poisson import *

from grids import Domain, Grid


class SolverTests(unittest.TestCase):

    def _prepare_gaussian_example(self):
        domain = Domain(center=(0, 0, 0), edges=(1, 1, 1))
        grid = Grid(domain, shape=(17, 17, 17))

        # Prepare exact solution

        def exact_solution(x, y, z):
            return math.exp(-(x ** 2 + y ** 2 + z ** 2))

        exact = grid.field_from_function(exact_solution)

        # Prepare problem to solve
        def the_rhs(x, y, z):
            r2 = x ** 2 + y ** 2 + z ** 2
            return (4 * r2 - 6) * math.exp(-r2)

        rhs = grid.field_from_function(the_rhs)

        bc = {}
        for ind in grid.boundary:
            x, y, z = grid.loc(ind)
            bc[ind] = exact_solution(x, y, z)

        return grid, rhs, bc, exact

    def _error_of_solution(self, exact, approx):
        return np.max(np.abs(exact.values - approx.values))


    @unittest.skip("")
    def test_simple_jacobi(self):

        grid, rhs, bc, exact = self._prepare_gaussian_example()
        solver = SimpleSolver(rhs, bc, method='jacobi')

        try:
            solver.solve()
        except Exception as e:
            self.fail()

        err = self._error_of_solution(exact, solver.solution())
        self.assertTrue(err < 2.0E-3)


    @unittest.skip("")
    def test_simple_gauss_seidl(self):

        grid, rhs, bc, exact = self._prepare_gaussian_example()
        solver = SimpleSolver(rhs, bc, method='gauss_seidel')

        try:
            solver.solve()
        except Exception as e:
            self.fail()

        err = self._error_of_solution(exact, solver.solution())
        self.assertTrue(err < 2.0E-3)


    #@unittest.skip("")
    def test_multigrid_solver(self):

        grid, rhs, bc, exact = self._prepare_gaussian_example()
        solver = MultiGridSolver(rhs, bc)

        try:
            solver.solve()
        except Exception as e:
            self.fail()

        err = self._error_of_solution(exact, solver.solution())
        self.assertTrue(err < 2.0E-3)


if __name__ == '__main__':
    unittest.main()