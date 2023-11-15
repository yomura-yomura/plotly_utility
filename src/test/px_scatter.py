import numpy as np
import plotly
import plotly.express as px

import plotly_utility.express as pux

if __name__ == "__main__":
    x = np.random.normal(size=100_000)
    y = np.random.normal(size=100_000)
    color = ["a"] * 50_000 + ["b"] * 50_000
    color = None
    px.scatter(
        x=x, y=y, color=color, marginal_x="histogram", marginal_y="histogram"
    ).show()
    pux.scatter(
        x=x, y=y, color=color, marginal_x="histogram", marginal_y="histogram"
    ).show()
