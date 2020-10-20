import numpy as np
import numpy_utility as npu
import warnings
import plotly.express as px
import plotly.graph_objs as go


__all__ = ["histogram"]


def histogram(
    data_frame=None,
    x=None,
    y=None,
    color=None,
    facet_row=None,
    facet_col=None,
    facet_col_wrap=0,
    facet_row_spacing=None,
    facet_col_spacing=None,
    hover_name=None,
    hover_data=None,
    animation_frame=None,
    animation_group=None,
    category_orders={},
    labels={},
    color_discrete_sequence=None,
    color_discrete_map={},
    marginal=None,
    opacity=None,
    orientation=None,
    barmode="relative",
    barnorm=None,
    histnorm=None,
    log_x=False,
    log_y=False,
    range_x=None,
    range_y=None,
    # histfunc=None,
    # cumulative=None,
    nbins=None,
    title=None,
    template=None,
    width=None,
    height=None,
):
    """
    Precomputing histogram binning in Python, not in Javascript.
    """
    args = px._core.build_dataframe(locals(), go.histogram)
    color = args["data_frame"][args["color"]] if args["color"] is not None else None

    if x is not None and y is None:
        x = args["data_frame"][args["x"]].to_numpy()
        orientation = "v"
    elif x is None and y is not None:
        x = args["data_frame"][args["y"]].to_numpy()
        orientation = "h"
    else:
        raise NotImplementedError

    if histnorm is None or histnorm == "":
        density = False
    elif histnorm == "probability density":
        density = True
    else:
        raise NotImplementedError(f"histnorm={histnorm} not supported yet")

    if nbins is None:
        _bins = "auto"
    else:
        _bins = nbins

    _a = np.array(x)
    if np.issubdtype(x.dtype, np.datetime64):
        if np.array(1, x.dtype).astype("M8[us]").view(int) == 0:
            warnings.warn(f"""
    histogram does not draw accurately using datetime objects with more precise unit than [us]:
        the maximum number of bins: {len(np.unique(_a))} -> {len(np.unique(_a.astype('M8[us]')))}
    For more accurate histogram, You should use px.histogram or lose the precise somehow.
            """)
        _a = _a.astype("M8[us]")

    if npu.is_floating(_a) and np.isnan(_a).any():
        not_nan = ~np.isnan(_a)
        warnings.warn(f"""
    Detected nan is removed: {_a.size} -> {np.count_nonzero(not_nan)}
        """)
        _a = _a[not_nan]
        if color is not None:
            color = np.array(color)[not_nan]

    _bins = npu.histogram_bin_edges(_a, bins=_bins)
    bin_width = npu.histogram_bin_widths(_bins)
    x = npu.histogram_bin_centers(_bins)
    if len(x) == 0:
        return px.bar()

    if color is None:
        y, _ = npu.histogram(_a, bins=_bins, density=density)
        _names = np.array([""] * len(_a))
    else:
        _names = np.array(color)
        y = np.array([
            y
            for name in np.unique(_names)
            for y in npu.histogram(_a[_names == name], bins=_bins, density=density)[0]
        ])
        color = np.array([
            name
            for name in np.unique(_names)
            for _ in x
        ])
        assert y.size % x.size == 0
        _n_unique_names = y.size // x.size
        x = np.tile(x, _n_unique_names)
        bin_width = np.tile(bin_width, _n_unique_names)

    if orientation == "h":
        x, y = y, x

    fig = px._core.make_figure(
        args=locals(),
        constructor=go.Bar,
        trace_patch=dict(
            textposition="auto",
            width=bin_width // 1000 if np.issubdtype(x.dtype, np.datetime64) else bin_width,
            marker_line_width=0,
            orientation=orientation
        ),
        layout_patch=dict(
            barmode=barmode,
            bargap=0
        )
    )

    for trace in fig.data:
        if np.all(bin_width != 1):
            if trace.yaxis == "y":  # Update Histograms
                trace.update(
                    hovertemplate=trace.hovertemplate.replace("%{x}", "%{customdata[0]} - %{customdata[1]}"),
                    customdata=np.c_[x - bin_width // 2, x + bin_width // 2]
                )
        elif np.any(bin_width != 1):
            warnings.warn("Not implemented in a variable-bin-width case")

        if marginal is not None and marginal != "":
            if trace.yaxis == "y2":  # Replace the binned marginal plots to the unbinned ones
                trace.x = _a[_names == trace.name]
    return fig
