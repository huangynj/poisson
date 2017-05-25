import matplotlib.pyplot as plt
import numpy as np

class Field:

    def __init__(self, grid):
        self.domain = grid.domain
        self.grid = grid
        self.values = np.zeros(grid.shape)

    def __getitem__(self, index):
        return (self.grid.loc(index), self.values[index])

    def quick_plot(self):
        axes = [self.grid.axis_as_indices(i) for i in [0,1,2] ]

        for i in range(3):
            axis = axes[i]
            coordinates = [ self.grid.location(ind)[i] for ind in axis ]
            values = [ self.values[ind] for ind in axis ]
            plt.plot(coordinates, values, 'x')

        plt.show()

