import numpy as np
import more_itertools
import plotly.express as px
import plotly.graph_objs as go


__all__ = ["get_uncertainty_trace"]


def get_uncertainty_trace(
        x, lower_y, upper_y, name=None, flip_xy=False,
        discrete_values=False, marker_color="#444444", opacity=0.3
):
    if name is None:
        name = f"Systematic Uncertainty Band"
    else:
        name = f"Systematic Uncertainty Band of {name}"

    if discrete_values:
        if len(x) - 1 != len(lower_y):
            raise ValueError("len(x) - 1 == len(lower_y)")
        x = list(more_itertools.roundrobin(x[:-1], x[1:]))
        lower_y = list(more_itertools.roundrobin(lower_y, lower_y))
        upper_y = list(more_itertools.roundrobin(upper_y, upper_y))

    x = np.asarray(x)
    lower_y = np.asarray(lower_y)
    upper_y = np.asarray(upper_y)

    mask = np.isnan(x) | np.isnan(lower_y) | np.isnan(upper_y)
    x = x[~mask]
    lower_y = lower_y[~mask]
    upper_y = upper_y[~mask]

    trace = go.Scatter(
        name=name,
        mode="lines",
        x=np.append(x, x[::-1]),
        y=np.append(lower_y, upper_y[::-1]),
        line=dict(width=0),
        fillcolor='rgba({}, {}, {}, {})'.format(*px.colors.hex_to_rgb(marker_color), opacity),
        fill='toself',
        hoverinfo="skip",
        showlegend=True,
        legendgroup="Systematic Uncertainty"
    )

    if flip_xy:
        x = trace.x
        y = trace.y
        trace.y = x
        trace.x = y

    return trace