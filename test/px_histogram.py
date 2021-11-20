import numpy as np
import plotly_utility.express as pux
from plotly_utility.offline import plot
import plotly.express as px
import plotly



if __name__ == "__main__":
    df = px.data.tips()

    common_kwargs = dict(
        x="total_bill",
        marginal="box",
        # color=color,
        barmode="overlay",
        # histnorm="probability density",
        # nbins=10,
        # labels={"x": "dimension 1"},
        # facet_row=color
        color="time",
        facet_col="sex",
        # facet_col_wrap=2,
        nbins=10,
        # weight="size"
    )

    plot([
        pux.histogram(df, title="plotly_utility.express.histogram", **common_kwargs),
        px.histogram(df, title="plotly.express.histogram", **common_kwargs)
    ])

    import pandas as pd
    df = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    df = df.sort_values(by="age")
    figs = list(pux.make_histograms_with_facet_col(df, x="survived", facet_col="age", as_qualitative=True))
    for f in figs:
        f.show()

    # # Datetime array
    # x -= np.min(x)
    # x *= 1000
    # x = x.astype(int).astype("M8[D]")
    #
    # common_kwargs = dict(
    #     x=x, marginal="box", color=color, barmode="overlay",
    #     # histnorm="probability density",
    #     # nbins=10,
    #     # labels={"x": "dimension 1"},
    # )
    #
    # plot([
    #     pux.histogram(title="plotly_utility.express.histogram", **common_kwargs),
    #     px.histogram(title="plotly.express.histogram", **common_kwargs)
    # ])
