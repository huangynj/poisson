# poisson
A lightweight multigrid solver for the 3D Poisson equation in Python

## Usage
Using the poisson reduces effectively to one method, `poisson.solve(rhs, bc)`.
Here `rhs` is a 3D numpy array holding the values of the right hand side of
the Poisson equation at the equidistant cartesian grid points. `bc` is a
Python dict holding the Dirichlet boundary condition values. The mapping is
from grid point index to value, e.g. for imposing a value 5 at the grid point
 with index `(0,2,3)` we have `bc[(0,2,3)] = 5`.


## Dependencies
The solver has dependencies to numpy and matplotlib. To install them, use pip:

```
pip install numpy
pip install matplotlib
```

