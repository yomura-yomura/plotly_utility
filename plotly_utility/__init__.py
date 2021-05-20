from . import graph_objects
from . import offline

__all__ = [
    "express", "graph_objects", "offline",

    "get_traces_at", "add_secondary_axis", "get_data",
    "to_numpy"
]

import plotly.graph_objects as go
import numpy as np
import numpy_utility as npu


def get_row_col(fig, xaxis, yaxis):
    table = npu.ja.apply(lambda ref: tuple(ref.trace_kwargs.values()), fig._grid_ref, 3).squeeze()
    matched = np.argwhere(np.all(table == (xaxis, yaxis), axis=-1))
    if len(matched) == 0:
        raise ValueError(f"xaxis '{xaxis}' and yaxis '{yaxis}' not found.")
    ret = (matched[0] + 1).tolist()
    if len(ret) == 1:
        return [1, ret[0]]
    elif len(ret) == 2:
        return ret
    else:
        raise NotImplementedError


def get_traces_at(fig: go.Figure, row=1, col=1):
    if fig._grid_ref is None:
        assert row is None and col is None
        return list(fig.data)
    else:
        grid_refs = np.array(
            [[list(c) for c in r] for r in fig._grid_ref],
            dtype=[("subplot_type", "U10"), ("layout_keys", object), ("trace_kwargs", object)]
        )
        n_row, n_col, n_data = grid_refs.shape
        if row == "all":
            row = np.arange(n_row) + 1
        if col == "all":
            col = np.arange(n_col) + 1

        assert np.isin(grid_refs["subplot_type"], ("xy", "scene")).all()

        x_anchors = [
            gr["trace_kwargs"]["xaxis"] for gr in grid_refs[row-1][col-1] if gr["subplot_type"] == "xy"
        ]
        y_anchors = [
            gr["trace_kwargs"]["yaxis"] for gr in grid_refs[row - 1][col - 1] if gr["subplot_type"] == "xy"
        ]
        scene_anchors = [
            gr["trace_kwargs"]["scene"] for gr in grid_refs[row - 1][col - 1] if gr["subplot_type"] == "scene"
        ]

        return [
            trace for trace in fig.data
            if (
                (
                    hasattr(trace, "xaxis") and
                    hasattr(trace, "yaxis") and
                    trace.xaxis in x_anchors and
                    trace.yaxis in y_anchors
                ) or (
                    hasattr(trace, "scene") and
                    trace.scene in scene_anchors
                )
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


def get_data(fig, i_data=1, reverse_along_row=True):
    n_row, n_col, n_data, *_ = np.shape(fig._grid_ref)
    assert 0 < i_data <= n_data

    traces = np.array([
        [
            matched_traces[i_data - 1] if len(matched_traces) >= i_data else None
            for matched_traces in (get_traces_at(fig, row, col) for col in range(1, n_col+1))
        ]
        for row in range(1, n_row+1)
    ])
    x, *other_unique_x = np.unique([t.x for trace_rows in traces for t in trace_rows if t], axis=0)
    assert len(other_unique_x) == 0

    mask = traces == None

    y = np.ma.empty((*mask.shape, x.size))
    y.mask = True
    y[~mask] = [t.y for trace_rows in traces for t in trace_rows if t]
    y.mask &= mask[..., np.newaxis]
    if reverse_along_row:
        y = y[::-1]
    return x, y


def to_numpy(fig, return_coords=False):
    traces = [[get_traces_at(fig, ir + 1, ic + 1) for ic, _ in enumerate(r)]
              for ir, r in enumerate(fig._grid_ref)]

    len_traces = npu.ja.apply(np.size, traces, -2)
    max_n_traces = np.max(len_traces)

    def traces_to_numpy_array(traces):
        return [
            (
                traces[i]["hovertemplate"].split("<br>")[0].split("=")[1] if traces[i]["hovertemplate"] is not None else "",
                traces[i].x,
                traces[i].y,
                traces[i].error_x.array if traces[i].error_x.array is not None else [],
                traces[i].error_y.array if traces[i].error_y.array is not None else []
            ) if i < len(traces) else ("", [], [], [], [])
            for i in range(max_n_traces)
        ]
    data = npu.ja.apply(traces_to_numpy_array, traces, -2)
    data = np.rec.fromarrays(
        np.rollaxis(data, -1),
        names=["facet_col", "x", "y", "error_x", "error_y"]
    ).view(np.ma.MaskedArray)

    data.mask = len_traces[..., np.newaxis] <= np.expand_dims(np.arange(max_n_traces), tuple(range(len_traces.ndim)))
    data["facet_col"].mask |= (data["facet_col"] == "")

    if hasattr(fig, "_fit_results"):
        import standard_fit as sf
        fr = npu.from_dict({"fit_result": sf.to_numpy(fig._fit_results)})
        if fr.shape[2] < data.shape[2]:
            new_fr = np.ma.empty(data.shape, fr.dtype)
            new_fr.mask = True
            new_fr[:, :, :fr.shape[2]] = fr
            fr = new_fr

        data = npu.ma.merge_arrays((data, fr))

    reverse_along_rows = True
    if reverse_along_rows:
        data = data[::-1]

    # data = np.swapaxes(npu.ma.from_jagged_array(npu.ma.apply(traces_to_numpy_array, traces)), -2, -1)

    if return_coords:
        coords = {
            "row": np.arange(data.shape[0]),
            "column": np.arange(data.shape[1]),
            "trace": np.arange(data.shape[2])
        }
        return data, coords
    else:
        return data
