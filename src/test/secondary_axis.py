import numpy as np
from plotly.offline import plot

import plotly_utility
import plotly_utility.express as pux

np.random.seed(0)

fig = pux.histogram(x=np.random.normal(size=1_000))
trace = plotly_utility.get_traces_at(fig)[0]
area = (trace.width * trace.y).sum()
# scale = 1/area*100

fig = pux.scatter(
    x=np.random.normal(size=1_000), y=np.random.normal(size=1_000), log_y=True
).update_traces(mode="lines+markers")
scale = 100

plot(fig)

plot(
    plotly_utility.add_secondary_axis(
        fig,
        1,
        1,
        anchor="y",
        scale=scale,
        secondary_axis_patch=dict(title="probability density", ticksuffix="%"),
    ),
    show_link=True,
)
