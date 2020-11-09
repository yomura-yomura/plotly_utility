import numpy as np
import plotly_utility.express as pux
from plotly_utility.offline import plot
import plotly.express as px
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    np.random.seed(0)
    # x = np.random.normal(size=1_000_000)
    x = np.random.normal(size=1_000_00)
    color = np.random.choice(["a", "b", "c", "d"], size=x.size)

    common_kwargs = dict(
        x=x,
        # marginal="box",
        # color=color,
        barmode="overlay",
        # histnorm="probability density",
        # nbins=10,
        # labels={"x": "dimension 1"},
        # facet_row=color
        facet_col=color,
        facet_col_wrap=2
    )

    plot([
        pux.histogram(title="plotly_utility.express.histogram", **common_kwargs),
        px.histogram(title="plotly.express.histogram", **common_kwargs)
    ])

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
