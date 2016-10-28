"""
Examples:
with figure_grid(5, 3) as grid:
  grid.next_subplot()
  # plot something
  grid.next_subplot()
  # plot something
  # ...etc

with figure_grid(10, 4) as grid:
  for i, axis in enumerate(grid.each_subplot()):
    # plot something
"""

import matplotlib.pyplot as plt

class figure_grid():
    def next_subplot(self, **kwargs):
        self.subplots += 1
        return self.fig.add_subplot(self.rows, self.cols, self.subplots, **kwargs)

    def each_subplot(self):
        for _ in range(self.rows * self.cols):
            yield self.next_subplot()

    def __init__(self, rows, cols, rowheight=4, rowwidth=10):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(rowwidth, rowheight*self.rows))
        self.subplots = 0

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        plt.tight_layout()
        plt.show()
