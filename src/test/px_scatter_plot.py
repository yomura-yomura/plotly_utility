import numpy as np
import pandas as pd
import plotly

import plotly_utility.express as pux

if __name__ == "__main__":
    np.random.seed(0)

    df = pd.DataFrame()
    size = 10_000
    df["x1"] = np.random.normal(0, 10, size=size)
    df["y1"] = 2 * df["x1"] + np.random.normal(0, 1, size=size)
    df["x2"] = np.random.normal(0, 1, size=size)
    df["y2"] = 0.5 * df["x2"] + np.random.normal(0, 1, size=size)
    df["type"] = np.random.choice(["A", "B"], size=size)

    fig = pux.scatter_matrix(df.drop(columns="type"))
    fig.show()

    fig = pux.scatter_matrix(df, color="type", barmode="overlay")
    fig.show()
