import plotly_utility.express as pux
import plotly_utility.subplots
import numpy as np
import pandas as pd
import numpy_utility as npu
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    x = np.arange(1024)
    y1 = 10 + 3 * (np.random.random(size=x.size) - 0.5)
    y2 = y1 + 5 * (np.random.random(size=x.size) - 0.5)
    y3 = y1 + 1 * (np.random.random(size=x.size) - 0.5)
    data = npu.to_tidy_data({
        "A": zip(x, y1),
        "B": zip(x, y2),
        "C": zip(x, y3),
    }, "type", ["x", "y"])
    fig = pux.scatter(data, x="x", y="y", color="type")
    fig = plotly_utility.subplots.add_residual_plot(fig, target_name="A")
    fig.show()
