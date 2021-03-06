import numpy as np
import pandas as pd
import plotly_utility.express as pux
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    np.random.seed(0)

    df = pd.DataFrame()
    size = 10_000
    df["x1"] = np.random.normal(0, 10, size=size)
    df["y1"] = 2 * df["x1"] + np.random.normal(0, 1, size=size)
    df["x2"] = np.random.normal(0, 1, size=size)
    df["y2"] = 0.5 * df["x2"] + np.random.normal(0, 1, size=size)

    fig = pux.scatter_matrix(df)
    fig.show()
