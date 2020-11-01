from . import graph_objects
from . import offline

__all__ = ["express", "graph_objects", "offline"]


import plotly.graph_objects as go
import numpy as np


def get_traces_at(fig: go.Figure, row=1, col=1):
    row -= 1
    col -= 1
    domain = fig._grid_ref[row][col]
    assert len(domain) == 1
    x_anchor = domain[0].trace_kwargs["xaxis"]
    y_anchor = domain[0].trace_kwargs["yaxis"]
    return [
        trace for trace in fig.data
        if (
            trace.xaxis == x_anchor if trace.xaxis is not None else "x" == x_anchor
        ) and (
            trace.yaxis == y_anchor if trace.yaxis is not None else "y" == y_anchor
        )
    ]


def add_secondary_axis(fig: go.Figure, row=1, col=1, i_data=1, anchor="x",
                       secondary_axis_patch: dict = None, scale=1, add_hover_text=True):
    import copy
    copied_trace = copy.deepcopy(get_traces_at(fig, row, col)[i_data - 1])

    if copied_trace.type == "box":
        raise NotImplementedError(f"{copied_trace.type} not implemented yet.")

    copied_trace.name = "_copied_" + copied_trace.name
    copied_trace.showlegend = False

    copied_trace.marker.color = "red"  # just for debugging

    if anchor not in ("x", "y"):
        raise ValueError(f"{anchor} must be either 'x' or 'y'.")

    def get_axis_names(fig, anchor):
        return [key for key in fig.layout if key.startswith(f"{anchor}axis")]

    if not hasattr(copied_trace, anchor):
        raise ValueError(f"{type(copied_trace)} has not an attribute '{anchor}'.")
    elif getattr(copied_trace, anchor) is None:
        raise ValueError(f"{anchor} is not defined in {type(copied_trace)}")

    # for efficiency, but it shows a strange behavior in log-scale.
    # x = getattr(copied_trace, anchor)
    # i_min = np.argmin(x)
    # i_min2 = np.argmin(x[x > 0])  # for log-scale
    # i_max = np.argmax(x)
    # if copied_trace.type == "scatter":
    #     copied_trace.x = [copied_trace.x[i_min], copied_trace.x[i_min2], copied_trace.x[i_max]]
    #     copied_trace.y = [copied_trace.y[i_min], copied_trace.y[i_min2], copied_trace.y[i_max]]
    # elif copied_trace.type == "bar":
    #     copied_trace.x = [copied_trace.x[i_min], copied_trace.x[i_min2], copied_trace.x[i_max]]
    #     copied_trace.y = [copied_trace.y[i_min], copied_trace.y[i_min2], copied_trace.y[i_max]]
    #     copied_trace.width = [0, 0]
    # else:
    #     raise NotImplementedError

    copied_trace.opacity = 0
    copied_trace.hovertemplate = None
    copied_trace.hoverinfo = "skip"

    setattr(copied_trace, anchor, np.array(getattr(copied_trace, anchor)) * scale)

    new_axis_number = len(get_axis_names(fig, anchor)) + 1
    this_axis_prefix = getattr(copied_trace, f"{anchor}axis")[1:]
    setattr(copied_trace, f"{anchor}axis", f"{anchor}{new_axis_number}")
    setattr(fig.layout, f"{anchor}axis{new_axis_number}", getattr(fig.layout, f"{anchor}axis{this_axis_prefix}"))
    secondary_axis = getattr(fig.layout, f"{anchor}axis{new_axis_number}")
    secondary_axis.title.text = None
    secondary_axis.overlaying = anchor
    secondary_axis.showgrid = False
    # secondary_axis.zeroline = False
    secondary_axis.side = "top" if anchor == "x" else "right"

    if secondary_axis_patch is not None:
        secondary_axis.update(secondary_axis_patch)

    fig.add_trace(copied_trace)

    if add_hover_text:
        trace = get_traces_at(fig, row, col)[i_data - 1]

        secondary_x = getattr(copied_trace, anchor)
        if trace.customdata is None:
            i = 0
            trace.customdata = np.c_[secondary_x]
        else:
            i = trace.customdata.shape[1]
            trace.customdata = np.c_[trace.customdata, secondary_x]

        var_name = secondary_axis.title.text if secondary_axis.title.text is not None else f"secondary {anchor}"
        x_suffix = secondary_axis.ticksuffix if secondary_axis.ticksuffix is not None else ""
        trace.hovertemplate += f"<br>{var_name}=%{{customdata[{i}]}}{x_suffix}"

    return fig

