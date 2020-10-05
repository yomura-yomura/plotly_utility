import numpy as np
import plotly_utility.express as pux
import plotly.express as px


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.normal(size=1_000_000)
    color = ["a"] * 500_000 + ["b"] * 500_000

    common_kwargs = dict(
        x=x, marginal="box", color=color, barmode="overlay",
        # histnorm="probability density",
        # nbins=10,
        # labels={"x": "dimension 1"},
    )

    if False:
        pux.histogram(title="plotly_utility.express.histogram", **common_kwargs).show()
        px.histogram(title="plotly.express.histogram", **common_kwargs).show()

    if not False:
        # Datetime array

        x -= np.min(x)
        x *= 1000
        x = x.astype(int).astype("M8[D]")

        common_kwargs = dict(
            x=x, marginal="box", color=color, barmode="overlay",
            # histnorm="probability density",
            # nbins=10,
            # labels={"x": "dimension 1"},
        )

        pux.histogram(title="plotly_utility.express.histogram", **common_kwargs).show()
        px.histogram(title="plotly.express.histogram", **common_kwargs).show()
