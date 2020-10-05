import plotly.express as px


__all__ = ["ridgeline_plot"]


def ridgeline_plot(x, y):
    """
    # x: (m) and y: (m),
    x: (m, n) and y: (m) or
    # x: (m, n) and y: (m, n)
    """
    x, y = zip(*[(x__, y_) for x_, y_ in zip(x, y) for x__ in x_])

    return px.violin(
        x=x,
        y=y,
        orientation="h",
    ).update_traces(side="positive", width=3)