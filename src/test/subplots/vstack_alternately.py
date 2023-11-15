import numpy as np
import plotly.express as px

import plotly_utility.express as pux
import plotly_utility.subplots

if __name__ == "__main__":
    np.random.seed(0)

    size = 1_000

    x = np.sqrt(np.random.normal(size=size) ** 2 + np.random.normal(size=size) ** 2)
    row = np.random.choice([1, 2], size=size)
    hist_fig = pux.histogram(x=x, facet_row=row)
    res_fig = px.box(x=x, facet_row=row)

    fig = plotly_utility.subplots.vstack_alternately(
        res_fig, hist_fig, small_spacing_fraction=0.1, vertical_spacing=0.3
    )
    fig.show()
