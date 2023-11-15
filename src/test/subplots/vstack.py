import numpy as np

import plotly_utility
import plotly_utility.express as pux

if __name__ == "__main__":
    fig = pux.histogram(x=np.random.normal(0, 1, size=1_000))
    # fig = plotly_utility.subplots.vstack(fig, fig)
    # fig = plotly_utility.subplots.hstack(fig, fig)
    # fig = plotly_utility.subplots.vstack(fig, fig)
    # fig = plotly_utility.subplots.hstack(fig, fig)

    fig1 = plotly_utility.subplots.hstack(fig, fig)
    fig2 = plotly_utility.subplots.vstack(fig1, fig)

    plotly_utility.subplots.vstack(fig2, fig2).show()

    fig2.show()
