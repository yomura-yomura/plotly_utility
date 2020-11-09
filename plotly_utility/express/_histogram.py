import numpy as np
import numpy_utility as npu
import warnings
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import plotly_utility


__all__ = ["histogram"]


possible_marginal_types = ["", "rug", "box", "violin", "histogram"]


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
    height=None
):
    """
    Precomputing histogram binning in Python, not in Javascript.
    """
    if marginal is None:
        pass
    elif marginal not in possible_marginal_types:
        raise ValueError(f"""
    Got invalid marginal '{marginal}.
    Possible marginal: {", ".join(possible_marginal_types)}
        """)

    args = px._core.build_dataframe(locals().copy(), go.Histogram)

    if args["histnorm"] is None or args["histnorm"] == "":
        density = False
    elif args["histnorm"] == "probability density":
        density = True
    else:
        raise NotImplementedError(f"histnorm={args['histnorm']} not supported yet")

    if args["nbins"] is None:
        bins = "auto"
    else:
        bins = args["nbins"]

    if (args["x"] is None) and (args["y"] is None):
        return px.bar()
    elif (args["x"] is not None) and (args["y"] is None):
        args["y"] = "y"
        swap_xy = False
    elif (args["x"] is None) and (args["y"] is not None):
        args["x"] = "x"
        args["x"], args["y"] = args["y"], args["x"]
        swap_xy = True
    else:
        raise NotImplementedError("Not supported yet in the case x and y are not None ")

    # args["data_frame"] = args["data_frame"].dropna(subset=[args["x"]])
    args["data_frame"] = args["data_frame"][np.isfinite(args["data_frame"][args["x"]].to_numpy())]
    data = args["data_frame"][args["x"]].to_numpy()

    if np.issubdtype(data.dtype, np.datetime64):
        x_converted = data.astype("M8[us]")
        if x_converted.size != data.size:
            warnings.warn(f"""
    histogram does not draw accurately using datetime objects with more precise unit than [us]:
        the maximum number of bins: {len(np.unique(data))} -> {len(np.unique(x_converted))}
    For more accurate histogram, You should use px.histogram or lose the precise somehow.
            """)
        data = x_converted

    bins = npu.histogram_bin_edges(data, bins=bins)
    bin_width = npu.histogram_bin_widths(bins)
    x = npu.histogram_bin_centers(bins)

    if len(data) == 0:
        return px.bar()

    use_one_plot = (args["color"] is None) and (args["facet_row"] is None) and (args["facet_col"] is None)

    if use_one_plot:
        y, _ = npu.histogram(data, bins=bins, density=density)
        data_classes = np.array([""] * len(data))
        args["data_frame"] = pd.DataFrame()
    else:
        groups = []
        if args["color"] is not None:
            groups.append(args["data_frame"][args["color"]])
        if args["facet_row"] is not None:
            groups.append(args["data_frame"][args["facet_row"]])
        if args["facet_col"] is not None:
            groups.append(args["data_frame"][args["facet_col"]])

        data_classes = np.array(np.transpose(groups).tolist())
        un = np.unique(data_classes, axis=0)
        y = np.array([
            y
            for name in un
            for y in npu.histogram(data[np.all(data_classes == name, axis=1)], bins=bins, density=density)[0]
        ])
        groups = np.array([
            name
            for name in un
            for _ in x
        ]).T.tolist()

        args["data_frame"] = pd.DataFrame()

        if args["facet_col"] is not None:
            args["data_frame"][args["facet_col"]] = groups.pop()
        if args["facet_row"] is not None:
            args["data_frame"][args["facet_row"]] = groups.pop()
        if args["color"] is not None:
            args["data_frame"][args["color"]] = groups.pop()

        assert y.size % x.size == 0
        _n_unique_names = y.size // x.size
        x = np.tile(x, _n_unique_names)
        bin_width = np.tile(bin_width, _n_unique_names)

    args["data_frame"][args["x"]] = x
    args["data_frame"][args["y"]] = y
    # print(args)

    if swap_xy:
        args["x"], args["y"] = args["y"], args["x"]
        args["orientation"] = "h"
        args["labels"] = {"x": "count"}
        target = "y"
    else:
        args["labels"] = {"y": "count"}
        target = "x"

    fig = px._core.make_figure(
        args=args,
        constructor=go.Bar,
        trace_patch=dict(
            textposition="auto",
            width=bin_width // 1000 if np.issubdtype(data.dtype, np.datetime64) else bin_width,
            marker_line_width=0,
            orientation=args["orientation"]
        ),
        layout_patch=dict(
            barmode=args["barmode"],
            bargap=0
        )
    )

    for trace in fig.data:
        if np.all(bin_width != 1):
            if trace.yaxis == "y":  # Update Histograms
                trace.update(
                    hovertemplate=trace.hovertemplate.replace(f"%{{{target}}}", "%{customdata[0]} - %{customdata[1]}"),
                    customdata=np.c_[x - bin_width // 2, x + bin_width // 2]
                )
        elif np.any(bin_width != 1):
            warnings.warn("Not implemented in a variable-bin-width case. Hover-data is disabled.")

    has_marginal = args["marginal"] is not None and args["marginal"] != ""
    if has_marginal:
        # marginal_traces = plotly_utility.get_traces_at(fig, 2, "all")
        marginal_traces = [t for t in fig.data if t.type in possible_marginal_types]
        if use_one_plot:
            assert len(marginal_traces) == 1
            marginal_traces[0].x = data
        else:
            assert len(marginal_traces) == len(un)
            for trace, n in zip(marginal_traces, un):
                # Replace the binned marginal plots to the unbinned ones
                trace.x = data[np.all(data_classes == n, axis=1)]

    return fig
