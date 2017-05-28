from grids import Domain, Grid
from poisson import MultiGridSolver

def g(x, y, z):
    """ Some example function used here to produce the boundary conditions """
    return x**3 + y**3 + z**3

def f(x, y, z):
    """ Some example function used here to produce the right hand side field """
    return 6*(x+y+z)

def example():
    """ This function demonstrates the usage of the poisson module and some of its classes """

    #
    # Make the grid on which to solve the Poisson equation
    #

    domain = Domain(center=(0,0,0), edges=(1,1,1))
    grid = Grid(domain, shape=(33,33,33))

    #
    # Prepare the boundary conditions
    #

    bc = {}
    for index in grid.boundary:
        x, y, z = grid.loc(index)
        bc[index] = g(x, y, z)

    #
    # Prepare the field with the right-hand-side of the Poisson equation.
    #

    rhs = grid.field_from_function(f)

        # rhs is of type Field, which has two properties: grid and values, the
        # latter being a numpy ndarray holding the field values at the grid
        # points

    #
    # Now solve the Poisson equation \Delta u = rhs
    #

    solver = MultiGridSolver(rhs, bc, atol=1.0E-6)

    try:
        solver.solve()
        u = solver.solution()  # u is of type Field

        # print solution

        for index in u.grid.indices():
            (x, y, z), u_val = u[index]
            print("{:10.2f}{:10.2f}{:10.2f}{:12.6f}".format(x, y, z, u_val))

    except Exception as e:
        print("No convergence")