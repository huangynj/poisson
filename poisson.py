""" Solve the Poisson equation on cartesian coordinates in 3 dimensions
    with Dirichlet boundary conditions.
    \Delta u  = f
"""

from grids import *


class Solver:
    """ The base class for the other solvers """
    
    def __init__(self, rhs, bc, atol=1.0E-6):
        self.rhs = rhs
        self.bc = bc
        self.atol = atol
        self.converged = False
        self.sol = rhs.grid.field()
        self.stepper = None
        self.max_steps = 10000
        self.impose_boundary_conditions()

    def impose_boundary_conditions(self):
        for key, value in self.bc.items():
            self.sol.values[key] = value

    def residual(self, sol, rhs):

        nx, ny, nz = sol.grid.shape
        dx, dy, dz = sol.grid.spacing()
        dxi2, dyi2, dzi2 = dx**-2, dy**-2, dz**-2

        u = sol.values
        f = rhs.values
        resid = sol.grid.field()
        r = resid.values

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    val = (u[i+1,j,k] -2*u[i,j,k] + u[i-1,j,k]) * dxi2 \
                                +(u[i,j+1,k] -2*u[i,j,k] + u[i,j-1,k]) * dyi2 \
                                +(u[i,j,k+1] -2*u[i,j,k] + u[i,j,k-1]) * dzi2 - f[i,j,k]
                    resid.values[(i, j, k)] = val
        return resid

    def solve(self):
        step_count = 0
        while True:
            step_count += 1
            self.step()
            err = self.check_convergence()
            print("{}, {:12.4e}".format(step_count, err))
            if self.converged or step_count == self.max_steps:
                break

        if not self.converged:
            raise Exception("No convergence")

    def check_convergence(self):
        resid = self.residual(self.sol, self.rhs)

        max_err = np.max(np.abs(resid.values))

        self.converged = False
        if max_err < self.atol:
            self.converged = True

        return max_err

    def solution(self):
        return self.sol



class SimpleSolver(Solver):
    """ A simple relaxation solver """
    
    def __init__(self, rhs, bc, method="jacobi", atol=1.0E-6):
        super(SimpleSolver, self).__init__(rhs, bc, atol)

        if method == "jacobi":
            self.stepper = JacobiStepper()
        elif method == "gauss_seidel":
            self.stepper = GaussSeidelStepper()
        else:
            raise Exception("No such stepper")

    def step(self):
        self.stepper.step(self.sol, self.rhs)


class MultiGridSolver(Solver):
    """ The actual multigrid solver """
    
    def __init__(self, rhs, bc, atol=1.0E-6):
        super(MultiGridSolver, self).__init__(rhs, bc, atol)

        self.multi_grid = MultiGrid(rhs.grid)
        self.pre_smooth_iter = 3
        self.post_smooth_iter = 3
        self.max_steps = 1000
        self.stepper = GaussSeidelStepper()

    def step(self):
        self.sol = self._do_multi_grid_step(self.sol, self.rhs)

    def _do_multi_grid_step(self, sol, rhs):

        self._do_smooth(sol, rhs, self.pre_smooth_iter)

        if self.multi_grid.has_coarser(sol.grid):
            resid = self.residual(sol, rhs)
            rhs_c = self.multi_grid.coarsify(resid)
            e_c = rhs_c.grid.field()
            e_c = self._do_multi_grid_step(e_c, rhs_c)
            sol_delta = self.multi_grid.refine(e_c)
            sol.values = sol.values - sol_delta.values

        self._do_smooth(sol, rhs, self.post_smooth_iter)
        return sol

    def _do_smooth(self, sol, rhs, num_iter):
        for it in range(num_iter):
            self.stepper.step(sol, rhs)


class JacobiStepper:
    """ Jacobi method:

     Discretized Poisson equation:

      (u[i+1,j,k] + u[i-1,j,k]) * dxi2
    + (u[i,j+1,k] + u[i,j-1,k]) * dyi2
    + (u[i,j,k+1] + u[i,j,k-1]) * dzi2
    - 2* (dxi2 + dyi2 + dzi2) u[i,j,k]
    + f[i,j,k]  = 0

      (u[i+1,j,k] + u[i-1,j,k]) * dxi2
    + (u[i,j+1,k] + u[i,j-1,k]) * dyi2
    + (u[i,j,k+1] + u[i,j,k-1]) * dzi2
    + f[i,j,k]  = 2* (dxi2 + dyi2 + dzi2) u[i,j,k]

    u[i,j,k] = (
          (u[i+1,j,k] + u[i-1,j,k]) * dxi2
        + (u[i,j+1,k] + u[i,j-1,k]) * dyi2
        + (u[i,j,k+1] + u[i,j,k-1]) * dzi2
        + f[i,j,k]
         ) / (2*(dxi2 + dyi2 + dzi2))

    """

    def step(self, field, rhs):
        u = field.values
        f = rhs.values

        nx, ny, nz = field.grid.shape
        dx, dy, dz = field.grid.spacing()
        dxi2, dyi2, dzi2 = 1./dx**2, 1./dy**2, 1./dz**2
        inv_denom = 1. / (2 * (dxi2 + dyi2 + dzi2))

        u_new = np.array(u)

        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    u_new[i,j,k] = (
                          (u[i+1,j,k] + u[i-1,j,k]) * dxi2
                        + (u[i,j+1,k] + u[i,j-1,k]) * dyi2
                        + (u[i,j,k+1] + u[i,j,k-1]) * dzi2
                        - f[i,j,k]
                         ) * inv_denom

        field.values = u_new


class GaussSeidelStepper:
    """ Gauss Seidel Method
    """

    def step(self, field, rhs):
        u = field.values
        f = rhs.values

        nx, ny, nz = field.grid.shape
        dx, dy, dz = field.grid.spacing()
        dxi2, dyi2, dzi2 = 1./dx**2, 1./dy**2, 1./dz**2
        inv_denom = 1. / (2*(dxi2 + dyi2 + dzi2))

        un = np.array(u)

        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    un[i,j,k] = (
                          (un[i+1,j,k] + un[i-1,j,k]) * dxi2
                        + (un[i,j+1,k] + un[i,j-1,k]) * dyi2
                        + (un[i,j,k+1] + un[i,j,k-1]) * dzi2
                        - f[i,j,k]
                         ) * inv_denom

        field.values = un


def solve(rhs, bc, edges):
    """ Interface to the outside world for those who do not want to use the module classes """
    
    domain = Domain(center=(0,0,0), edges=edges)
    grid = Grid(domain, tuple(rhs.shape))
    field_rhs = Field(grid)
    field_rhs.values = np.array(rhs)
    solver = MultiGridSolver(field_rhs, bc)
    return solver.solve().solution()
